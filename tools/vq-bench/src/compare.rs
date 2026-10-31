//! Baseline-versus-candidate comparison, ported from ds4-perf's compare.rs.
//!
//! The arithmetic is the same on purpose: timings are compared in seconds per
//! token (1 / tps), the point estimate is the median of the samples, and the
//! band is the worst-case envelope formed from the extrema of both sides, so a
//! change is only called real when the two runs do not overlap even at their
//! most favourable pairing. That is stricter than comparing medians, and it is
//! what makes a small VQ change distinguishable from run-to-run noise.

use crate::csv::Row;

#[derive(Clone, Debug)]
pub struct Delta {
    pub metric: String,
    pub ctx: u64,
    pub baseline_median: f64,
    pub candidate_median: f64,
    pub median_percent: f64,
    pub lower_percent: f64,
    pub upper_percent: f64,
}

pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let middle = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[middle - 1] + v[middle]) / 2.0
    } else {
        v[middle]
    }
}

pub fn delta(metric: &str, ctx: u64, a: &[f64], b: &[f64]) -> Result<Delta, String> {
    if a.is_empty() || b.is_empty() || a.iter().chain(b).any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("nonpositive or missing timing samples".into());
    }
    let baseline_median = median(a);
    let candidate_median = median(b);
    let extrema = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(0.0f64, f64::max),
        )
    };
    let (amin, amax) = extrema(a);
    let (bmin, bmax) = extrema(b);
    Ok(Delta {
        metric: metric.into(),
        ctx,
        baseline_median,
        candidate_median,
        median_percent: 100.0 * (candidate_median / baseline_median - 1.0),
        lower_percent: 100.0 * (bmin / amax - 1.0),
        upper_percent: 100.0 * (bmax / amin - 1.0),
    })
}

/// Group by ctx and compare the three timing metrics. The unit is seconds per
/// token (or seconds for first_token), as in ds4-perf.
pub fn compare(a: &[Row], b: &[Row]) -> Result<Vec<Delta>, String> {
    let mut out = Vec::new();
    let mut ctxs: Vec<u64> = a.iter().map(|r| r.ctx).collect();
    ctxs.sort_unstable();
    ctxs.dedup();
    ctxs.retain(|c| b.iter().any(|r| r.ctx == *c));

    for ctx in ctxs {
        let pick = |rows: &[Row]| -> Vec<Row> { rows.iter().filter(|r| r.ctx == ctx).cloned().collect() };
        let (ra, rb) = (pick(a), pick(b));
        let per_token = |rows: &[Row], f: fn(&Row) -> f64| -> Vec<f64> { rows.iter().map(|r| 1.0 / f(r)).collect() };
        out.push(delta("prefill", ctx, &per_token(&ra, |r| r.prefill_tps), &per_token(&rb, |r| r.prefill_tps))?);
        out.push(delta("gen", ctx, &per_token(&ra, |r| r.gen_tps), &per_token(&rb, |r| r.gen_tps))?);
        out.push(delta("first_token", ctx, &ra.iter().map(|r| r.first_token_sec).collect::<Vec<_>>(),
                       &rb.iter().map(|r| r.first_token_sec).collect::<Vec<_>>())?);
    }
    Ok(out)
}
