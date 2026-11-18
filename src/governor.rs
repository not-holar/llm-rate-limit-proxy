//! Model-pressure governor: a per-model concurrency gate beside the rate limiter.
//!
//! Providers cap how many requests may be *in flight* on a model at once, which
//! is a different resource from the per-key request rate and is shared across
//! every key. Failing over to another key cannot relieve it, so the usual
//! response to a rebuff — bench the lane, try the next one — burns healthy key
//! capacity on an attempt that was never going to work.
//!
//! Detection is behavioral rather than a string match on the provider's error.
//! Per-key rate limiters are independent, and the pool already paces each key
//! under its own budget, so rebuffs on one key say nothing about another. When
//! the *same model* is rebuffed on two different keys within seconds, that
//! correlation is the evidence: the constraint is model-scoped. This needs no
//! knowledge of the provider's error wording, so it keeps working when that
//! wording changes and works across providers that phrase it differently.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Rebuffs older than this are no longer evidence of anything current.
const EVIDENCE_WINDOW: Duration = Duration::from_secs(10);

/// Distinct keys that must be rebuffed on one model before we hold the model
/// responsible. Two suffices: independent limiters do not correlate by chance.
const KEYS_TO_INDICT: usize = 2;

/// A governed model with no rebuff for this long returns to ungoverned. The
/// provider's worker pool is shared infrastructure, so the real ceiling moves
/// with other tenants' load and a cap we picked minutes ago is stale.
const DISSOLVE_AFTER: Duration = Duration::from_secs(10 * 60);

/// Minimum gap between two cuts. One episode of pressure produces a burst of
/// rebuffs — every request already in flight fails at once — and halving on each
/// of them ratchets the cap toward 1 in seconds. Each cut lowers concurrency,
/// which lowers the observed in-flight count, which makes the next cut smaller
/// still: a spiral that ends with the model throttled far below what the
/// provider would actually allow. One episode, one cut.
const ADJUST_COOLDOWN: Duration = Duration::from_secs(5);

/// Never cap a model below this. One permit serializes every caller behind a
/// single request, which on a slow model is indistinguishable from an outage —
/// and a provider that cannot serve two at once has a problem no gate fixes.
const MIN_LIMIT: usize = 2;

/// Additive increase: a model with no rebuff for this long gets one more permit.
/// Cutting hard and climbing back slowly is what stops a transient blip from
/// parking a model at a cap it no longer needs.
///
/// Growth is evaluated lazily, when a request asks for a permit, so this is one
/// permit per minute *of traffic* rather than of wall clock. Under load that is
/// the same thing; on an idle model nothing grows, which costs nothing because
/// an idle model is not waiting on the cap, and a long enough silence dissolves
/// it outright. Measured live: a model cut to 20 sat at 20 while it drained,
/// then rose to 21.
const GROW_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Default)]
struct ModelState {
    /// Concurrency cap; 0 means ungoverned.
    limit: usize,
    /// Permits currently held — requests in flight upstream on this model.
    inflight: usize,
    /// Recent rebuffs as (when, which lane), pruned to `EVIDENCE_WINDOW`.
    rebuffs: Vec<(Instant, usize)>,
    last_rebuff: Option<Instant>,
    /// When the cap last moved, in either direction. Paces both cuts and growth.
    last_adjusted: Option<Instant>,
}

impl ModelState {
    fn prune(&mut self, now: Instant) {
        self.rebuffs
            .retain(|(at, _)| now.duration_since(*at) < EVIDENCE_WINDOW);
    }

    /// How many distinct keys have been rebuffed inside the evidence window.
    fn implicated_keys(&self) -> usize {
        let mut lanes: Vec<usize> = self.rebuffs.iter().map(|(_, lane)| *lane).collect();
        lanes.sort_unstable();
        lanes.dedup();
        lanes.len()
    }
}

/// A model under a concurrency cap, as reported to operators.
#[derive(serde::Serialize, Debug, PartialEq)]
pub struct ModelPressure {
    pub model: String,
    pub limit: usize,
    pub inflight: usize,
}

#[derive(Default)]
pub struct Governor {
    models: Mutex<HashMap<String, ModelState>>,
}

impl Governor {
    /// Record that `model` was rebuffed while being served by `lane`.
    ///
    /// Once two distinct keys are implicated the model is governed, starting at
    /// half the concurrency we were actually running. Halving is deliberate: the
    /// true ceiling is unknown and shared with other tenants, so the useful move
    /// is to back off decisively and climb back, not to guess a number.
    pub async fn note_rebuff(&self, model: &str, lane: usize, now: Instant) {
        let mut models = self.models.lock().await;
        let state = models.entry(model.to_string()).or_default();
        state.prune(now);
        state.rebuffs.push((now, lane));
        state.last_rebuff = Some(now);

        // One episode of pressure fails every in-flight request at once. Cutting
        // on each of those rebuffs would ratchet the cap toward 1 within seconds.
        let just_adjusted = state
            .last_adjusted
            .is_some_and(|at| now.duration_since(at) < ADJUST_COOLDOWN);

        // A rebuff that lands while we are already under our own cap says nothing
        // about the cap. Our gate was not the constraint, so the pressure came
        // from elsewhere on a pool we share — and cutting for that lets one busy
        // neighbour ratchet us to a standstill. Measured: without this the cap
        // walked 75 -> 3 -> 1 over five minutes while throughput decayed with it.
        let our_gate_was_binding = state.limit == 0 || state.inflight >= state.limit;

        if state.implicated_keys() >= KEYS_TO_INDICT && !just_adjusted && our_gate_was_binding {
            let observed = state.inflight.max(1);
            let backed_off = (observed / 2).max(MIN_LIMIT);
            let was = state.limit;
            state.limit = if state.limit == 0 {
                backed_off
            } else {
                state.limit.min(backed_off)
            };
            if state.limit != was {
                state.last_adjusted = Some(now);
                // The one moment worth announcing: from here requests wait on a
                // gate that consumes no rate budget, so every rate figure will
                // read low while the proxy is in fact working as hard as allowed.
                println!(
                    "  governing {model} at {} concurrent ({} keys rebuffed, {} in flight)",
                    state.limit,
                    state.implicated_keys(),
                    observed
                );
            }
        }
    }

    /// Current concurrency cap for `model`; 0 means ungoverned. A model that has
    /// been quiet for `DISSOLVE_AFTER` releases its cap here.
    pub async fn limit(&self, model: &str, now: Instant) -> usize {
        let mut models = self.models.lock().await;
        let Some(state) = models.get_mut(model) else {
            return 0;
        };
        if state
            .last_rebuff
            .is_some_and(|at| now.duration_since(at) >= DISSOLVE_AFTER)
        {
            state.limit = 0;
            state.rebuffs.clear();
            state.last_rebuff = None;
            state.last_adjusted = None;
            return 0;
        }
        // Additive increase. Without a way back up, one bad moment parks the
        // model at a reduced cap until it dissolves entirely — throttling it for
        // minutes on evidence that is seconds old.
        if state.limit > 0 {
            let since = state.last_adjusted.or(state.last_rebuff);
            if since.is_some_and(|at| now.duration_since(at) >= GROW_INTERVAL) {
                state.limit += 1;
                state.last_adjusted = Some(now);
            }
        }
        state.limit
    }

    /// Every model currently under a cap, with its cap and live concurrency.
    ///
    /// This is the operator-facing answer to "why is it slow when the keys look
    /// idle". A request gated here never reaches the rate limiter, so rate-based
    /// capacity reads 0% while everything stalls; without this the blocking
    /// constraint is invisible.
    pub async fn pressured(&self, now: Instant) -> Vec<ModelPressure> {
        let mut models = self.models.lock().await;
        let mut out: Vec<ModelPressure> = models
            .iter_mut()
            .filter_map(|(model, state)| {
                if state
                    .last_rebuff
                    .is_some_and(|at| now.duration_since(at) >= DISSOLVE_AFTER)
                {
                    state.limit = 0;
                    state.rebuffs.clear();
                    state.last_rebuff = None;
                }
                (state.limit > 0).then(|| ModelPressure {
                    model: model.clone(),
                    limit: state.limit,
                    inflight: state.inflight,
                })
            })
            .collect();
        out.sort_by(|a, b| a.model.cmp(&b.model));
        out
    }

    /// Requests in flight on `model` right now.
    pub async fn inflight(&self, model: &str) -> usize {
        self.models
            .lock()
            .await
            .get(model)
            .map_or(0, |s| s.inflight)
    }
}

/// Claim a concurrency slot on `model`, or `None` when it is at its cap.
/// Ungoverned models always admit. The permit releases on drop.
pub async fn admit(gov: &Arc<Governor>, model: &str, now: Instant) -> Option<ModelPermit> {
    let limit = gov.limit(model, now).await;
    let mut models = gov.models.lock().await;
    let state = models.entry(model.to_string()).or_default();
    if limit > 0 && state.inflight >= limit {
        return None;
    }
    state.inflight += 1;
    Some(ModelPermit {
        governor: gov.clone(),
        model: model.to_string(),
    })
}

/// Holds one model-concurrency slot, releasing it on drop so a request frees it
/// however it ends.
pub struct ModelPermit {
    governor: Arc<Governor>,
    model: String,
}

impl Drop for ModelPermit {
    fn drop(&mut self) {
        if let Some(state) = self
            .governor
            .models
            .try_lock()
            .unwrap()
            .get_mut(&self.model)
        {
            state.inflight = state.inflight.saturating_sub(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn governor() -> Arc<Governor> {
        Arc::new(Governor::default())
    }

    #[tokio::test]
    async fn a_model_starts_ungoverned() {
        let g = governor();
        assert_eq!(g.limit("m", Instant::now()).await, 0);
    }

    /// The discriminator that makes string matching unnecessary: one key being
    /// rebuffed over and over is a lane problem, and benching that lane is the
    /// right answer. Governing the model here would throttle every healthy key
    /// because one key is sick.
    #[tokio::test]
    async fn repeated_rebuffs_on_one_key_do_not_indict_the_model() {
        let g = governor();
        let now = Instant::now();
        for _ in 0..10 {
            g.note_rebuff("m", 0, now).await;
        }
        assert_eq!(g.limit("m", now).await, 0);
    }

    /// Independent per-key limiters do not fail together by chance, so the same
    /// model being rebuffed on two different keys within seconds is evidence the
    /// constraint is the model — no knowledge of the error wording required.
    #[tokio::test]
    async fn rebuffs_across_distinct_keys_indict_the_model() {
        let g = governor();
        let now = Instant::now();
        g.note_rebuff("m", 0, now).await;
        assert_eq!(g.limit("m", now).await, 0, "one key is not yet evidence");
        g.note_rebuff("m", 1, now).await;
        assert!(
            g.limit("m", now).await > 0,
            "two distinct keys indicts the model"
        );
    }

    #[tokio::test]
    async fn evidence_older_than_the_window_is_not_counted() {
        let g = governor();
        let start = Instant::now();
        g.note_rebuff("m", 0, start).await;
        // The first rebuff has aged out by the time the second key fails, so the
        // two never coexist as evidence and the model stays ungoverned.
        let later = start + EVIDENCE_WINDOW + Duration::from_secs(1);
        g.note_rebuff("m", 1, later).await;
        assert_eq!(g.limit("m", later).await, 0);
    }

    #[tokio::test]
    async fn one_models_pressure_does_not_govern_another() {
        let g = governor();
        let now = Instant::now();
        g.note_rebuff("busy", 0, now).await;
        g.note_rebuff("busy", 1, now).await;
        assert!(g.limit("busy", now).await > 0);
        assert_eq!(g.limit("quiet", now).await, 0, "pressure is per model");
    }

    #[tokio::test]
    async fn a_governed_model_admits_up_to_its_cap_then_refuses() {
        let g = governor();
        let now = Instant::now();
        // Four in flight when the pressure shows up, so the cap lands at two.
        let held: Vec<_> = (0..4).map(async |_| admit(&g, "m", now).await).collect();
        g.note_rebuff("m", 0, now).await;
        g.note_rebuff("m", 1, now).await;
        assert_eq!(g.limit("m", now).await, 2);
        drop(held);

        let _a = admit(&g, "m", now).await.expect("first fits under the cap");
        let _b = admit(&g, "m", now)
            .await
            .expect("second fits under the cap");
        assert!(admit(&g, "m", now).await.is_none(), "third exceeds the cap");
    }

    #[tokio::test]
    async fn a_permit_frees_its_slot_on_drop() {
        let g = governor();
        let now = Instant::now();
        let held: Vec<_> = (0..2).map(async |_| admit(&g, "m", now).await).collect();
        g.note_rebuff("m", 0, now).await;
        g.note_rebuff("m", 1, now).await;
        drop(held);
        assert_eq!(g.inflight("m").await, 0);
        assert!(
            admit(&g, "m", now).await.is_some(),
            "freed slots are reusable"
        );
    }

    #[tokio::test]
    async fn an_ungoverned_model_admits_without_limit() {
        use tokio_stream::StreamExt;

        let g = governor();
        let now = Instant::now();
        let held: Vec<_> = tokio_stream::iter(0..64)
            .then(|_| admit(&g, "m", now))
            .collect()
            .await;
        assert!(held.iter().all(|p: &Option<ModelPermit>| p.is_some()));
    }

    #[tokio::test]
    async fn pressured_reports_only_governed_models() {
        let g = governor();
        let now = Instant::now();
        let _quiet = admit(&g, "quiet", now).await;
        assert!(
            g.pressured(now).await.is_empty(),
            "nothing is under pressure yet"
        );

        g.note_rebuff("busy", 0, now).await;
        g.note_rebuff("busy", 1, now).await;
        let reported = g.pressured(now).await;
        assert_eq!(reported.len(), 1, "only the governed model is reported");
        assert_eq!(reported[0].model, "busy");
        assert!(reported[0].limit > 0);
    }

    /// Observed live against NVIDIA: one episode of pressure fails every
    /// in-flight request at once, and cutting on each rebuff ratcheted a model
    /// 6 → 5 → 4 → 3 → 1 concurrent in seconds. Each cut shrinks in-flight,
    /// which shrinks the next cut — a spiral to 1 that no amount of provider
    /// capacity can escape.
    #[tokio::test]
    async fn a_burst_of_rebuffs_from_one_episode_cuts_only_once() {
        use tokio_stream::StreamExt;

        let g = governor();
        let now = Instant::now();
        let _held: Vec<_> = tokio_stream::iter(0..8)
            .map(|_| admit(&g, "m", now))
            .collect()
            .await;
        for lane in 0..8 {
            g.note_rebuff("m", lane, now).await;
        }
        assert_eq!(
            g.limit("m", now).await,
            4,
            "eight rebuffs from one episode must halve once, not eight times"
        );
    }

    /// Measured live over five minutes: a cap already down at 3 kept being cut
    /// on rebuffs that arrived with only 2 in flight, reaching 1 and staying
    /// there while throughput decayed 252 -> 12 rpm.
    ///
    /// A rebuff that arrives while we are *under* our own cap is not evidence
    /// the cap is too generous — our gate was not the constraint. The provider's
    /// pool is shared, so that pressure is somebody else's load, and cutting for
    /// it means one busy neighbour can ratchet us to a standstill.
    #[tokio::test]
    async fn pressure_below_the_cap_does_not_cut_further() {
        use tokio_stream::StreamExt;

        let g = governor();
        let start = Instant::now();
        let held: Vec<_> = tokio_stream::iter(0..8)
            .then(|_| admit(&g, "m", start))
            .collect()
            .await;
        g.note_rebuff("m", 0, start).await;
        g.note_rebuff("m", 1, start).await;
        assert_eq!(
            g.limit("m", start).await,
            4,
            "first cut halves observed concurrency"
        );
        drop(held);

        // Now only two requests are in flight — well under the cap of 4.
        let _two: Vec<_> = tokio_stream::iter(0..2)
            .then(|_| admit(&g, "m", start))
            .collect()
            .await;
        let later = start + ADJUST_COOLDOWN + Duration::from_secs(1);
        g.note_rebuff("m", 2, later).await;
        g.note_rebuff("m", 3, later).await;
        assert_eq!(
            g.limit("m", later).await,
            4,
            "our gate was not the constraint, so the cap must hold"
        );
    }

    #[tokio::test]
    async fn a_cap_never_falls_below_the_floor() {
        use tokio_stream::StreamExt;

        let g = governor();
        let mut now = Instant::now();
        // Drive it as hard as possible: always rebuff at exactly the cap.
        for round in 0..12 {
            let limit = g.limit("m", now).await.max(1);
            let held: Vec<_> = tokio_stream::iter(0..limit)
                .then(async |_| admit(&g, "m", now).await)
                .filter(|x| x.is_some())
                .collect()
                .await;
            g.note_rebuff("m", round * 2, now).await;
            g.note_rebuff("m", round * 2 + 1, now).await;
            drop(held);
            now += ADJUST_COOLDOWN + Duration::from_secs(1);
        }
        assert!(
            g.limit("m", now).await >= MIN_LIMIT,
            "collapsed to {}",
            g.limit("m", now).await
        );
    }

    #[tokio::test]
    async fn a_stable_model_climbs_back_one_permit_at_a_time() {
        use tokio_stream::StreamExt;

        let g = governor();
        let start = Instant::now();
        let held: Vec<_> = tokio_stream::iter(0..8)
            .then(|_| admit(&g, "m", start))
            .collect()
            .await;
        g.note_rebuff("m", 0, start).await;
        g.note_rebuff("m", 1, start).await;
        drop(held);
        assert_eq!(g.limit("m", start).await, 4);

        let after_one = start + GROW_INTERVAL + Duration::from_secs(1);
        assert_eq!(
            g.limit("m", after_one).await,
            5,
            "one stable interval, one permit"
        );
        let after_two = after_one + GROW_INTERVAL + Duration::from_secs(1);
        assert_eq!(g.limit("m", after_two).await, 6);
    }

    #[tokio::test]
    async fn growth_does_not_resurrect_an_ungoverned_model() {
        let g = governor();
        let now = Instant::now();
        let much_later = now + GROW_INTERVAL * 10;
        assert_eq!(g.limit("never-seen", much_later).await, 0);
    }

    #[tokio::test]
    async fn a_quiet_model_returns_to_ungoverned() {
        let g = governor();
        let start = Instant::now();
        g.note_rebuff("m", 0, start).await;
        g.note_rebuff("m", 1, start).await;
        assert!(g.limit("m", start).await > 0);
        let much_later = start + DISSOLVE_AFTER + Duration::from_secs(1);
        assert_eq!(
            g.limit("m", much_later).await,
            0,
            "the provider's ceiling moves with other tenants; a stale cap is wrong"
        );
    }
}
