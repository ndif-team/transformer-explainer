import { writable, derived, readable, get } from 'svelte/store';
import tailwindConfig from '../../tailwind.config';
import resolveConfig from 'tailwindcss/resolveConfig';
import { ex0 } from '~/constants/examples';
import { textPages } from '~/utils/textbookPages';
import type { ServerModelMetadata } from '~/types/forwardPass';

const { theme } = resolveConfig(tailwindConfig);

export const attentionHeadIdxTemp = writable(0);
export const attentionHeadIdx = writable(0);
export const blockIdxTemp = writable(0);
export const blockIdx = writable(0);
export const isOnBlockTransition = writable(false);

export const isOnAnimation = writable(false);

// Textbook state management
export const textbookCurrentPage = writable<number>(0);
export const textbookPreviousPage = writable<number>(-1);
export const textbookCurrentPageId = writable<string>(textPages[0].id);
export const textbookPreviousPageId = writable<string>('');
export const isTextbookOpen = writable<boolean>(true);

// is transformer running?
export const isModelRunning = writable(false);
export const isFetchingModel = writable(true);
export const isLoaded = writable(false);

export const inputTextExample = [
	'Data visualization empowers users to',
	'Artificial Intelligence is transforming the',
	'As the spaceship was approaching the',
	'On the deserted planet they discovered a',
	'IEEE VIS conference highlights the'
];

const initialExIdx = 0;
export const selectedExampleIdx = writable<number>(initialExIdx);

// transformer model output
export const modelData = writable<ModelData>(ex0);
export const predictedToken = writable<Probability>();
export const tokens = writable<string[]>(ex0?.tokens);
export const tokenIds = writable<number[]>(ex0?.tokenIds);

// Server-supplied list of NDIF-hosted models, fetched on app start.
export const availableModels = writable<ServerModelMetadata[]>([]);

// Lookup of arch metadata by model name. Always seeded with GPT-2 defaults so
// the cached example fixtures render correctly before /models resolves; the
// fetchModels() bootstrap step replaces this with real server data.
export const modelMetaMap = writable<Record<string, ModelMetaData>>({
	gpt2: { layer_num: 12, attention_head_num: 12, dimension: 768 },
	'openai-community/gpt2': { layer_num: 12, attention_head_num: 12, dimension: 768 }
});

// selected token vector
export const highlightedToken = writable<HighlightedToken>({
	index: null,
	value: null,
	fix: false
});

export const highlightedHead = writable<HighlightedToken>({
	index: null,
	value: null,
	fix: false
});

// expanded block
export const expandedBlock = writable<ExpandedBlock>({ id: null });
export const isExpandOrCollapseRunning = writable(false);

// user input text
export const inputText = writable(inputTextExample[initialExIdx]);
// export const tokens = derived(inputText, ($inputText) => $inputText.trim().split(' '));

// selected model and meta data
const initialSelectedModel = 'gpt2';
export const selectedModel = writable<string>(initialSelectedModel);
export const modelMeta = derived(
	[selectedModel, modelMetaMap],
	([$selectedModel, $modelMetaMap]) => $modelMetaMap[$selectedModel] ?? $modelMetaMap['gpt2']
);

/** Hydrate modelMetaMap from a server model list. */
export function applyServerModels(models: ServerModelMetadata[]) {
	availableModels.set(models);
	const next: Record<string, ModelMetaData> = { ...get(modelMetaMap) };
	for (const m of models) {
		next[m.name] = {
			layer_num: m.n_layers,
			attention_head_num: m.n_heads,
			dimension: m.d_model,
			d_head: m.d_head,
			n_kv_heads: m.n_kv_heads,
			vocab_size: m.vocab_size,
			positional_kind: m.positional_kind,
			arch_kind: m.arch_kind
		};
	}
	modelMetaMap.set(next);
}

// Temperature setting
export const initialTemperature = 0.8;
export const temperature = writable(initialTemperature);

// Sampling
export const sampling = writable<Sampling>({ type: 'top-k', value: 5 });

// Prediction visual
export const highlightedIndex = writable(null);
export const finalTokenIndex = writable(null);

// Visual element style
export const rootRem = 16;
export const minVectorHeight = 12;
export const maxVectorHeight = 30;
export const maxVectorScale = 3.4;

export const vectorHeight = writable(0);
export const headContentHeight = writable(0);
export const headGap = { x: 5, y: 8, scale: 0 };

export const isBoundingBoxActive = writable(false);

export const predictedColor = theme.colors.purple[600];

// Interactivity
export const hoveredPath = writable();
export const hoveredMatrixCell = writable({ row: null, col: null });
export const weightPopover = writable();
export const tooltip = writable();

export const isMobile = readable(false, (set) => {
	if (typeof window !== 'undefined') {
		// Only run in browser environment
		const userAgent = navigator.userAgent.toLowerCase();
		set(/android|iphone|ipad|ipod/i.test(userAgent));
	}
	return () => {}; // Cleanup function
});

// User identification
export const userId = writable<string | null>(null);