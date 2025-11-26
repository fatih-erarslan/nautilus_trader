import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    // WASM entry commented out until wasm-pkg is built
    // 'wasm/index': 'src/wasm/index.ts',
  },
  format: ['cjs', 'esm'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  minify: false,
  treeshake: true,
  target: 'es2020',
  external: ['@neural-trader/predictor-native'],
});
