import { ClerkProvider } from "@clerk/nextjs";
import type { Metadata, Viewport } from "next";
import { headers } from "next/headers";
import { Nunito } from "next/font/google";

import { ExitModal } from "@/components/modals/exit-modal";
import { HeartsModal } from "@/components/modals/hearts-modal";
import { PracticeModal } from "@/components/modals/practice-modal";
import { Toaster } from "@/components/ui/sonner";
import { siteConfig } from "@/config";

import "./globals.css";

const font = Nunito({ subsets: ["latin"] });

export const viewport: Viewport = {
  themeColor: "#22C55E",
};

export const metadata: Metadata = siteConfig;

/**
 * The clone ships Clerk as its auth provider. FlyLingo's lesson is local and needs no
 * accounts, so Clerk is mounted only when a real publishable key is configured.
 *
 * The original layout wrapped every route unconditionally, which means a missing or
 * placeholder key replaced the entire app with Clerk's "Invalid host" error. Branching
 * here keeps the auth-native routes working when a key exists and lets the local lesson
 * run when one does not.
 */
const publishableKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY ?? "";
const keyIsReal =
  publishableKey.startsWith("pk_") &&
  !publishableKey.includes("ZXhhbXBsZ") &&
  !publishableKey.includes("example.clerk");

/**
 * Routes that must render without an auth provider.
 *
 * The FlyLingo surfaces are local and account-free, and an absent or placeholder Clerk key
 * makes ClerkProvider replace the whole tree with an error, so these routes skip it
 * entirely. The clone's own pages still get Clerk when a real key is configured.
 */
const CLERK_FREE_PREFIXES = ["/lesson/fly", "/fly"];

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Routes that do not use auth must not be wrapped in ClerkProvider: an absent or
  // placeholder key makes Clerk replace the whole tree with an error. The proxy
  // forwards the request path so this decision can be made here, on the server.
  const requestHeaders = await headers();
  const pathname = requestHeaders.get("x-pathname") ?? "";
  const clerkEnabled =
    keyIsReal && !CLERK_FREE_PREFIXES.some((p) => pathname.startsWith(p));

  const shell = (
    <html lang="en">
      <body className={font.className}>
        <Toaster theme="light" richColors closeButton />
        <ExitModal />
        <HeartsModal />
        <PracticeModal />
        {children}
      </body>
    </html>
  );

  if (!clerkEnabled) return shell;

  return (
    <ClerkProvider
      appearance={{
        options: {
          logoImageUrl: "/favicon.ico",
        },
        variables: {
          colorPrimary: "#22C55E",
        },
      }}
      telemetry={false}
      afterSignOutUrl="/"
    >
      {shell}
    </ClerkProvider>
  );
}
