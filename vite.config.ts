import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter({ fallback: undefined }),
			// Spaces serve the app at /index.html; hash routing ignores the path.
			router: { type: 'hash' }
		})
	],
	optimizeDeps: { exclude: ['onnxruntime-web'] }
});
