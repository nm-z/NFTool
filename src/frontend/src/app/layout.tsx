import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ApiProvider } from "@/components/ApiProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NFTool",
  description: "Advanced Neural Framework for Regression",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <meta
          httpEquiv="Content-Security-Policy"
          content="default-src 'self' tauri:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: http://localhost:8001 http://127.0.0.1:* https://placehold.co; connect-src 'self' http://localhost:8001 http://127.0.0.1:* ws://localhost:8001 ws://127.0.0.1:* ws://localhost:3000 http://localhost:3000; font-src 'self' data:;"
        />
      </head>
      <body
        className={`${inter.className} antialiased`}
        suppressHydrationWarning
      >
        <ApiProvider>
          {children}
        </ApiProvider>
      </body>
    </html>
  );
}
