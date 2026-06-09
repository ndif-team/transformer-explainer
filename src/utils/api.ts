import { writable } from 'svelte/store';
import type { ForwardPassData, ServerModelMetadata } from '~/types/forwardPass';

const BASE = (import.meta.env.VITE_WORKBENCH_API ?? 'http://localhost:8000').replace(/\/$/, '');
const USER_EMAIL = import.meta.env.VITE_USER_EMAIL ?? 'dev@localhost';

const HEADERS = {
	'Content-Type': 'application/json',
	'X-User-Email': USER_EMAIL
};

/** Latest NDIF job status, exposed as a store for the loading UI. */
export const jobStatus = writable<string>('idle');

export interface ForwardPassRequest {
	model: string;
	prompt: string;
	positions?: number[] | 'all';
	top_k?: number;
}

const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 120_000;

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
	const res = await fetch(url, { credentials: 'omit', ...init });
	if (!res.ok) {
		let detail = '';
		try {
			const body = await res.json();
			detail = body?.detail ?? body?.message ?? JSON.stringify(body);
		} catch {
			detail = await res.text().catch(() => '');
		}
		throw new Error(`${res.status} ${res.statusText}: ${detail}`);
	}
	return (await res.json()) as T;
}

async function pollUntilDone(jobId: string): Promise<void> {
	const started = Date.now();
	while (Date.now() - started < POLL_TIMEOUT_MS) {
		const status = await fetchJson<{ status: string; description?: string | null }>(
			`${BASE}/forward_pass/status/${jobId}`,
			{ method: 'GET', headers: HEADERS }
		);
		const s = status.status?.toUpperCase?.() ?? 'UNKNOWN';
		jobStatus.set(s);
		if (s === 'COMPLETED') return;
		if (s === 'ERROR' || s === 'NNSIGHT_ERROR') {
			throw new Error(`NDIF job ${s}: ${status.description ?? 'no details'}`);
		}
		await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
	}
	jobStatus.set('TIMEOUT');
	throw new Error('Forward-pass job timed out');
}

export async function runForwardPass(req: ForwardPassRequest): Promise<ForwardPassData> {
	jobStatus.set('STARTING');
	const body = JSON.stringify({
		model: req.model,
		prompt: req.prompt,
		positions: req.positions ?? [-1],
		top_k: req.top_k ?? 10
	});
	const startResp = await fetchJson<{ job_id?: string; data?: ForwardPassData }>(
		`${BASE}/forward_pass/start`,
		{ method: 'POST', headers: HEADERS, body }
	);

	if (startResp.data) {
		jobStatus.set('idle');
		return startResp.data;
	}
	if (!startResp.job_id) {
		throw new Error('Unexpected /forward_pass/start response (no job_id or data)');
	}

	await pollUntilDone(startResp.job_id);

	const resultsResp = await fetchJson<{ data: ForwardPassData }>(
		`${BASE}/forward_pass/results/${startResp.job_id}`,
		{ method: 'POST', headers: HEADERS, body }
	);
	jobStatus.set('idle');
	return resultsResp.data;
}

export async function fetchModels(): Promise<ServerModelMetadata[]> {
	const data = await fetchJson<ServerModelMetadata[]>(`${BASE}/models/`, {
		method: 'GET',
		headers: HEADERS
	});
	return data;
}
