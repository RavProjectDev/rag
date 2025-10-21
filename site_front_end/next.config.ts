import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  eslint: {
    // Do not fail the build on ESLint errors
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Do not fail the build on TypeScript errors
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
