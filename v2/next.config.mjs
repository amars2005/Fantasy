/** @type {import('next').NextConfig} */
const nextConfig = {
  outputFileTracingRoot: process.cwd(),

  // In production the bundle normally comes from BUNDLE_BASE_URL and nothing
  // needs to ship with the build. A bundle staged into `v2/data/v2_export`
  // (`npm run bundle:stage`) is the alternative for a deployment with no Blob
  // store, and tracing is what actually gets those files into the function --
  // nothing imports them, so they are otherwise invisible to the build.
  outputFileTracingIncludes: {
    "/api/**/*": ["./data/v2_export/**/*.json"],
    "/league/**/*": ["./data/v2_export/**/*.json"],
    "/": ["./data/v2_export/**/*.json"],
  },
};

export default nextConfig;
