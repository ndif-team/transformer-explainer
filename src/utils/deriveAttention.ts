/**
 * Derive the causally-masked + softmax-normalized attention views from a raw
 * `scores` matrix. The workbench backend only sends `scores` (post-RoPE,
 * pre-mask, pre-softmax) over the wire — the masked and softmax variants are
 * computed here so each attention tensor only has to be transmitted once
 * instead of three times. See nnsightful.tools.forward_pass for the
 * server-side counterpart.
 *
 * Both functions take a single head's score matrix of shape [S][S] and
 * return a same-shape matrix. The caller iterates heads.
 */

/**
 * Apply the causal mask by replacing every upper-triangle cell with 0.
 * This matches the legacy `_attn_masked` view the UI rendered when the
 * server used to send `scores_masked` (which itself substituted 0 for
 * −∞ to keep the JSON serializable).
 */
export function applyCausalMaskZero(scoresHead: number[][]): number[][] {
	const S = scoresHead.length;
	const out: number[][] = new Array(S);
	for (let q = 0; q < S; q++) {
		const row = scoresHead[q];
		const masked = new Array<number>(S);
		for (let k = 0; k < S; k++) {
			masked[k] = k > q ? 0 : row[k];
		}
		out[q] = masked;
	}
	return out;
}

/**
 * Apply causal mask + numerically-stable per-row softmax. Upper-triangle
 * cells (k > q) emerge as exactly 0 in the result; the lower-triangle
 * cells of each row sum to ≈ 1. Mirrors what the server used to compute
 * via torch.softmax(scores.masked_fill(triu, -inf), dim=-1).
 */
export function computeAttentionProbs(scoresHead: number[][]): number[][] {
	const S = scoresHead.length;
	const out: number[][] = new Array(S);
	for (let q = 0; q < S; q++) {
		const row = scoresHead[q];
		// Find max over the unmasked (k <= q) range for stability.
		let max = -Infinity;
		for (let k = 0; k <= q; k++) {
			if (row[k] > max) max = row[k];
		}
		// exp + sum over the unmasked range; everything else stays 0.
		const exps = new Array<number>(S);
		let sum = 0;
		for (let k = 0; k < S; k++) {
			if (k > q) {
				exps[k] = 0;
			} else {
				const e = Math.exp(row[k] - max);
				exps[k] = e;
				sum += e;
			}
		}
		// Normalize.
		for (let k = 0; k < S; k++) {
			exps[k] = exps[k] / sum;
		}
		out[q] = exps;
	}
	return out;
}
