import type { ForwardPassData } from '~/types/forwardPass';
import { applyCausalMaskZero, computeAttentionProbs } from './deriveAttention';

/**
 * Translate a ForwardPassData payload into the legacy `outputs` map keyed by
 * `block_{layer}_attn_head_{head}_attn[_masked|_softmax|_dropout]`. This lets
 * existing components (AttentionMatrix.svelte etc.) keep their selectors
 * unchanged while we migrate piecemeal to consume ForwardPassData directly.
 *
 * The backend only ships raw `scores`. The masked and softmax views are
 * derived here on the way in — see deriveAttention.ts for the math.
 */
export function adaptForwardPass(data: ForwardPassData): Record<string, { data: number[][] }> {
	const outputs: Record<string, { data: number[][] }> = {};
	data.layers.forEach((layer, layerIdx) => {
		const H = layer.attention.scores.length;
		for (let h = 0; h < H; h++) {
			const raw = layer.attention.scores[h];
			const masked = applyCausalMaskZero(raw);
			const softmax = computeAttentionProbs(raw);
			outputs[`block_${layerIdx}_attn_head_${h}_attn`] = { data: raw };
			outputs[`block_${layerIdx}_attn_head_${h}_attn_scaled`] = { data: raw };
			outputs[`block_${layerIdx}_attn_head_${h}_attn_masked`] = { data: masked };
			outputs[`block_${layerIdx}_attn_head_${h}_attn_softmax`] = { data: softmax };
			// "dropout" view in the legacy UI = post-softmax (no actual dropout at inference time).
			outputs[`block_${layerIdx}_attn_head_${h}_attn_dropout`] = { data: softmax };
		}
	});
	return outputs;
}
