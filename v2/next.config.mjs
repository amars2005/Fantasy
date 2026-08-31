/** @type {import('next').NextConfig} */
const nextConfig = {
  // The bundle is read off disk in development (`data/v2_export`) and from
  // Blob in production, so nothing needs bundling into the server build.
  outputFileTracingRoot: process.cwd(),
};

export default nextConfig;
