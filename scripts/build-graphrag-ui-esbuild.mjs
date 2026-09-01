#!/usr/bin/env node
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const repoRoot = path.resolve(path.dirname(__filename), "..");
const frontendRoot = path.join(repoRoot, "frontend");
const requireFromFrontend = createRequire(path.join(frontendRoot, "package.json"));
const esbuild = requireFromFrontend("esbuild");
const outdirArgIndex = process.argv.indexOf("--outdir");
const outdir = path.resolve(
  frontendRoot,
  outdirArgIndex >= 0 && process.argv[outdirArgIndex + 1]
    ? process.argv[outdirArgIndex + 1]
    : "dist-graphrag",
);

const navigationShim = path.join(
  frontendRoot,
  "src/config/graphragStandaloneNavigation.ts",
);
const srcRoot = path.join(frontendRoot, "src");

const SOURCE_EXTENSIONS = [".tsx", ".ts", ".jsx", ".js", ".json"];

const resolveSourcePath = (targetPath) => {
  if (path.extname(targetPath) && existsSync(targetPath)) return targetPath;

  for (const extension of SOURCE_EXTENSIONS) {
    const candidate = `${targetPath}${extension}`;
    if (existsSync(candidate)) return candidate;
  }

  for (const extension of SOURCE_EXTENSIONS) {
    const candidate = path.join(targetPath, `index${extension}`);
    if (existsSync(candidate)) return candidate;
  }

  return targetPath;
};

const urlImportPlugin = {
  name: "url-import",
  setup(build) {
    build.onResolve({ filter: /\?url$/ }, (args) => {
      const requestPath = args.path.replace(/\?url$/, "");
      const resolvedPath =
        requestPath.startsWith(".") || path.isAbsolute(requestPath)
          ? path.isAbsolute(requestPath)
            ? requestPath
            : path.resolve(args.resolveDir, requestPath)
          : requireFromFrontend.resolve(requestPath);
      return {
        path: resolvedPath,
        namespace: "url-file",
      };
    });

    build.onLoad({ filter: /.*/, namespace: "url-file" }, async (args) => ({
      contents: await readFile(args.path),
      loader: "file",
      resolveDir: path.dirname(args.path),
    }));
  },
};

const appAliasPlugin = {
  name: "app-alias",
  setup(build) {
    build.onResolve({ filter: /^@\/config\/navigation$/ }, () => ({
      path: navigationShim,
    }));

    build.onResolve({ filter: /^@\// }, (args) => ({
      path: resolveSourcePath(path.join(srcRoot, args.path.slice(2))),
    }));
  },
};

await rm(outdir, { recursive: true, force: true });
await mkdir(outdir, { recursive: true });

const result = await esbuild.build({
  absWorkingDir: frontendRoot,
  bundle: true,
  chunkNames: "assets/chunks/[name]-[hash]",
  define: {
    "import.meta.env": JSON.stringify({
      DEV: false,
      MODE: "production",
      PROD: true,
      VITE_API_BASE_URL: "",
      VITE_BACKEND_URL: "",
    }),
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  entryNames: "assets/[name]",
  entryPoints: ["src/graphragStandalone.tsx"],
  format: "esm",
  loader: {
    ".bmp": "file",
    ".gif": "file",
    ".jpeg": "file",
    ".jpg": "file",
    ".png": "file",
    ".svg": "file",
    ".ttf": "file",
    ".webp": "file",
    ".woff": "file",
    ".woff2": "file",
  },
  mainFields: ["browser", "module", "main"],
  metafile: true,
  minify: true,
  outdir,
  platform: "browser",
  plugins: [appAliasPlugin, urlImportPlugin],
  publicPath: "/",
  sourcemap: true,
  splitting: true,
  target: ["es2020"],
  treeShaking: true,
  write: true,
});

const outputs = Object.keys(result.metafile.outputs);
const entryScript = outputs.find((output) =>
  output.endsWith("/assets/graphragStandalone.js"),
);
const cssFiles = outputs.filter((output) => output.endsWith(".css"));

if (!entryScript) {
  throw new Error("GraphRAG standalone entry was not emitted.");
}

const entryScriptPath = `/${path.relative(outdir, path.resolve(frontendRoot, entryScript)).replaceAll(path.sep, "/")}`;
const cssLinks = cssFiles
  .map((output) => {
    const href = `/${path.relative(outdir, path.resolve(frontendRoot, output)).replaceAll(path.sep, "/")}`;
    return `    <link rel="stylesheet" href="${href}" />`;
  })
  .join("\n");

const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>GraphRAG Document Chat</title>
    <meta name="description" content="GraphRAG document upload, evidence, and chat workspace" />
${cssLinks ? `${cssLinks}\n` : ""}  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="${entryScriptPath}"></script>
  </body>
</html>
`;

await writeFile(path.join(outdir, "index.html"), html);

console.log(`Built GraphRAG standalone UI at ${outdir}`);
