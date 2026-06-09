import { get } from 'svelte/store';
import {
	modelData,
	tokens,
	tokenIds,
	isModelRunning,
	predictedToken,
	selectedModel
} from '~/store';
import { showFlowAnimation } from './animation';
import { runForwardPass, type ForwardPassRequest } from './api';
import { adaptForwardPass } from './adaptForwardPass';
import type { ForwardPassData, TopKLogits } from '~/types/forwardPass';

/** Map of token id -> decoded string surface form. */
type DecodeMap = Map<number, string>;

/** Display top-k/top-p sampling from a server-returned TopKLogits payload. */
export const getProbabilitiesFromTopK = ({
	topk,
	sampling,
	temperature
}: {
	topk: TopKLogits;
	sampling: Sampling;
	temperature: number;
}): { probabilities: Probabilities; sampled: Probability } => {
	const decode: DecodeMap = new Map(topk.token_ids.map((id, idx) => [id, topk.tokens[idx]]));
	return sampling.type === 'top-p'
		? topPSampling(topk, decode, sampling.value, temperature)
		: topKSampling(topk, decode, sampling.value, temperature);
};

function topKSampling(
	topk: TopKLogits,
	decode: DecodeMap,
	k: number,
	temperature: number
): { probabilities: Probabilities; sampled: Probability } {
	const items = topk.token_ids.map((tokenId, idx) => ({
		tokenId,
		logit: topk.logits[idx],
		scaledLogit: topk.logits[idx] / Math.max(temperature, 1e-6)
	}));

	const filtered = items.map((item, index) => ({
		...item,
		topKLogit: index < k ? item.scaledLogit : -Infinity
	}));

	const { expLogits, probabilities } = softmax(filtered.map((f) => f.topKLogit));

	const output: Probabilities = filtered.map((item, i) => ({
		...item,
		rank: i,
		token: formatTokenForDisplay(decode.get(item.tokenId) ?? `<id:${item.tokenId}>`),
		expLogit: expLogits[i],
		probability: probabilities[i]
	}));

	return { probabilities: output, sampled: randomChoice(output) };
}

function topPSampling(
	topk: TopKLogits,
	decode: DecodeMap,
	p: number,
	temperature: number
): { probabilities: Probabilities; sampled: Probability } {
	const items = topk.token_ids.map((tokenId, idx) => ({
		tokenId,
		logit: topk.logits[idx],
		scaledLogit: topk.logits[idx] / Math.max(temperature, 1e-6)
	}));
	const { expLogits, probabilities } = softmax(items.map((it) => it.scaledLogit));

	const cumulative: number[] = [];
	probabilities.reduce((acc, prob, idx) => {
		cumulative[idx] = acc + prob;
		return cumulative[idx];
	}, 0);

	let cutoff = cumulative.findIndex((cp) => cp >= p);
	cutoff = cutoff === -1 ? cumulative.length - 1 : cutoff;

	const topMask = items.map((_, i) => i <= cutoff);
	const sumTop = probabilities.filter((_, i) => topMask[i]).reduce((s, v) => s + v, 0);

	const output: Probabilities = items.map((item, i) => ({
		...item,
		rank: i,
		token: formatTokenForDisplay(decode.get(item.tokenId) ?? `<id:${item.tokenId}>`),
		expLogit: expLogits[i],
		probability: topMask[i] ? probabilities[i] / sumTop : 0,
		topPProbability: probabilities[i],
		cumulativeProbability: cumulative[i],
		cutoffIndex: cutoff
	}));

	return { probabilities: output, sampled: randomChoice(output) };
}

function softmax(logits: number[]): { expLogits: number[]; probabilities: number[] } {
	const finite = logits.filter((l) => l !== -Infinity);
	const maxLogit = finite.length ? Math.max(...finite) : 0;
	const expLogits = logits.map((logit) => (logit === -Infinity ? 0 : Math.exp(logit - maxLogit)));
	const sumExpLogits = expLogits.reduce((sum, val) => sum + val, 0) || 1;
	const probabilities = expLogits.map((val) => val / sumExpLogits);
	return { expLogits, probabilities };
}

function formatTokenForDisplay(token: string): string {
	return token
		.replace(/\n/g, '[NEWLINE]')
		.replace(/\t/g, '[TAB]')
		.replace(/\r/g, '[CR]')
		.replace(/\s{2,}/g, (match) => `[${match.length} SPACES]`);
}

function randomChoice(items: Probabilities): Probability {
	const random = Math.random();
	let cumulativeProbability = 0;
	for (let i = 0; i < items.length; i++) {
		cumulativeProbability += items[i].probability;
		if (random < cumulativeProbability) return items[i];
	}
	return items[items.length - 1];
}

/** Re-sample the displayed probabilities without re-running the model. */
export const adjustTemperature = async ({
	temperature,
	sampling
}: {
	temperature: number;
	sampling: Sampling;
}) => {
	const current = get(modelData);
	const topk = (current as ModelData & { __topk?: TopKLogits }).__topk;
	if (!topk) return;
	const { probabilities, sampled } = getProbabilitiesFromTopK({ topk, sampling, temperature });
	modelData.update((d) => ({ ...d, probabilities, sampled }));
	predictedToken.set(sampled);
};

/** Render cached example data — used for the initial paint before the API responds. */
export const fakeRunWithCachedData = async ({
	cachedData,
	temperature,
	sampling
}: {
	cachedData: ModelData & { tokens?: string[]; tokenIds?: number[]; __topk?: TopKLogits };
	temperature: number;
	sampling: Sampling;
}) => {
	isModelRunning.set(true);
	modelData.set(cachedData);
	if (cachedData.tokens) tokens.set(cachedData.tokens);
	if (cachedData.tokenIds) tokenIds.set(cachedData.tokenIds);
	setTimeout(async () => {
		await showFlowAnimation(cachedData.tokens?.length ?? 0, true);
		if (cachedData.__topk) {
			adjustTemperature({ temperature, sampling });
		}
		isModelRunning.set(false);
	}, 0);
};

/**
 * Drive a forward pass through the workbench /forward_pass endpoint, then
 * adapt the response into the legacy ModelData shape consumed by the
 * attention/mlp/embedding components.
 */
export const runModel = async ({
	input,
	temperature,
	sampling
}: {
	input: string;
	temperature: number;
	sampling: Sampling;
}) => {
	const model = get(selectedModel);
	if (!model) {
		console.warn('runModel called without a selected model');
		return;
	}
	isModelRunning.set(true);
	const request: ForwardPassRequest = {
		model,
		prompt: input === '' ? ' ' : input,
		positions: [-1],
		top_k: 50
	};

	let data: ForwardPassData;
	try {
		data = await runForwardPass(request);
	} catch (err) {
		console.error('forward_pass failed:', err);
		isModelRunning.set(false);
		throw err;
	}

	tokens.set(data.input_tokens);
	tokenIds.set(data.input_token_ids);

	const outputs = adaptForwardPass(data);
	const { probabilities, sampled } = getProbabilitiesFromTopK({
		topk: data.next_token,
		sampling,
		temperature
	});

	// Embed the top-k payload so adjustTemperature can re-sample without
	// another round-trip.
	const next: ModelData & { __topk?: TopKLogits; __arch?: ForwardPassData['meta']['arch'] } = {
		// Keep `logits` as a number[] for legacy reads; use the top-k logits.
		logits: data.next_token.logits,
		outputs: outputs as ModelData['outputs'],
		probabilities,
		sampled,
		__topk: data.next_token,
		__arch: data.meta.arch
	};
	modelData.set(next);

	setTimeout(async () => {
		await showFlowAnimation(data.input_tokens.length, false);
		predictedToken.set(sampled);
		isModelRunning.set(false);
	}, 0);
};
