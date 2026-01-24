import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",  // Enable static HTML export for bundling with PyInstaller
  images: {
    unoptimized: true,  // Required for static export (no image optimization server)
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8001",
        pathname: "/**",
      },
      {
        protocol: "https",
        hostname: "placehold.co",
        pathname: "/**",
      },
    ],
  },
};

export default nextConfig;
