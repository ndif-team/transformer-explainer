<script lang="ts">
	import { availableModels, selectedModel, isModelRunning, modelMeta } from '~/store';
	import classNames from 'classnames';
	import { onMount } from 'svelte';

	let open = false;
	let rootEl: HTMLDivElement;

	$: archBadge = $modelMeta?.arch_kind ?? 'unknown';

	const onPick = (name: string) => {
		selectedModel.set(name);
		open = false;
	};

	const handleDocClick = (e: MouseEvent) => {
		if (!open) return;
		if (rootEl && e.target instanceof Node && !rootEl.contains(e.target)) {
			open = false;
		}
	};

	const handleEsc = (e: KeyboardEvent) => {
		if (e.key === 'Escape') open = false;
	};

	onMount(() => {
		document.addEventListener('click', handleDocClick);
		document.addEventListener('keydown', handleEsc);
		return () => {
			document.removeEventListener('click', handleDocClick);
			document.removeEventListener('keydown', handleEsc);
		};
	});
</script>

<div class="model-selector relative" bind:this={rootEl} data-testid="model-selector">
	<button
		type="button"
		class={classNames(
			'flex items-center gap-2 rounded border border-gray-300 bg-white px-3 py-1 text-sm shadow-sm hover:bg-gray-50',
			$isModelRunning && 'cursor-not-allowed opacity-60'
		)}
		disabled={$isModelRunning}
		on:click|stopPropagation={() => (open = !open)}
		data-testid="model-selector-button"
	>
		<span class="font-mono text-gray-700" data-testid="model-selector-current">
			{$selectedModel ?? '—'}
		</span>
		<span
			class={classNames(
				'rounded px-1 text-[10px] uppercase tracking-wide',
				archBadge === 'gpt2' && 'bg-blue-100 text-blue-700',
				archBadge === 'gptj' && 'bg-purple-100 text-purple-700',
				archBadge === 'llama' && 'bg-orange-100 text-orange-700',
				archBadge === 'other' && 'bg-gray-100 text-gray-700'
			)}
			data-testid="model-selector-arch"
		>
			{archBadge}
		</span>
		<svg class="h-3 w-3 text-gray-500" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
			<path d="M5 8l5 5 5-5H5z" />
		</svg>
	</button>

	{#if open && $availableModels.length > 0}
		<ul
			class="model-selector-list absolute right-0 mt-1 max-h-80 w-64 overflow-auto rounded border border-gray-200 bg-white py-1 text-sm shadow-md"
			role="listbox"
			data-testid="model-selector-list"
		>
			{#each $availableModels as model (model.name)}
				<li>
					<button
						type="button"
						class={classNames(
							'flex w-full items-center justify-between gap-2 px-3 py-1 text-left hover:bg-gray-50',
							model.name === $selectedModel && 'bg-gray-100',
							model.allowed === false && 'cursor-not-allowed opacity-50'
						)}
						disabled={model.allowed === false}
						on:click|stopPropagation={() => onPick(model.name)}
						data-testid={`model-selector-option-${model.arch_kind}`}
					>
						<span class="font-mono text-xs text-gray-700">{model.name}</span>
						<span
							class={classNames(
								'rounded px-1 text-[10px] uppercase tracking-wide',
								model.arch_kind === 'gpt2' && 'bg-blue-100 text-blue-700',
								model.arch_kind === 'gptj' && 'bg-purple-100 text-purple-700',
								model.arch_kind === 'llama' && 'bg-orange-100 text-orange-700',
								model.arch_kind === 'other' && 'bg-gray-100 text-gray-700'
							)}>{model.arch_kind}</span
						>
					</button>
				</li>
			{/each}
		</ul>
	{/if}
</div>

<style lang="scss">
	.model-selector {
		flex-shrink: 0;
	}
	.model-selector-list {
		// Sit above the textbook / sankey / popover layers, which use z-indices
		// up to 1000 (see src/styles/variables.scss). Tailwind's z-50 is far too
		// low — the dropdown was opening, but every scene layer covered it.
		z-index: 1500;
	}
</style>
