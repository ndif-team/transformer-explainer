// Types mirroring nnsightful.types.ForwardPassData and friends.
// Source of truth: /home/researcher/nnsightful/src/nnsightful/types.py

export type ArchKind = 'gpt2' | 'llama' | 'gptj';
export type PositionalKind = 'absolute' | 'rope';
export type ArchKindClient = ArchKind | 'other';

export interface ForwardPassArch {
	kind: ArchKind;
	n_layers: number;
	n_heads: number;
	n_kv_heads: number;
	d_model: number;
	d_head: number;
	vocab_size: number;
	positional_kind: PositionalKind;
	has_fused_qkv: boolean;
	tie_word_embeddings: boolean;
}

export interface ForwardPassMeta {
	version: number;
	model: string;
	arch: ForwardPassArch;
}

export interface TopKLogits {
	token_ids: number[];
	tokens: string[];
	logits: number[];
	probs: number[];
}

export interface AttentionPayload {
	scores: number[][][]; // [head][q][k]
	scores_masked: number[][][];
	probs: number[][][];
}

export interface LayerPositionPayload {
	resid_pre: number[][];
	ln1_out: number[][];
	q: number[][][];
	k: number[][][];
	v: number[][][];
	attn_out: number[][];
	resid_mid: number[][];
	ln2_out: number[][];
	mlp_out: number[][];
	resid_post: number[][];
}

export interface LayerPayload {
	attention: AttentionPayload;
	per_position: LayerPositionPayload;
}

export interface ForwardPassData {
	meta: ForwardPassMeta;
	input_token_ids: number[];
	input_tokens: string[];
	positions: number[];
	tok_embed: number[][] | null;
	pos_embed: number[][] | null;
	input_embed: number[][] | null;
	layers: LayerPayload[];
	ln_final_out: number[][];
	topk_per_position: TopKLogits[];
	next_token: TopKLogits;
}

// Server-reported model metadata (from workbench /models endpoint).
export interface ServerModelMetadata {
	name: string;
	is_chat: boolean;
	n_layers: number;
	params: string;
	gated: boolean;
	allowed?: boolean;
	n_heads: number;
	n_kv_heads: number;
	d_model: number;
	d_head: number;
	vocab_size: number;
	positional_kind: PositionalKind;
	arch_kind: ArchKindClient;
}
