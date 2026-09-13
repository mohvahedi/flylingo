import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * FlyLingo proxy (Next 16's rename of middleware).
 *
 * The clone shipped `clerkMiddleware()` here for auth. FlyLingo's lesson runs locally and
 * needs no accounts, so auth middleware is dropped and replaced with something the app
 * actually needs: the request path, forwarded to the root layout.
 *
 * The layout uses it to decide whether to mount `<ClerkProvider />`. Without that branch a
 * missing or placeholder Clerk key replaces the entire app with Clerk's "Invalid host"
 * error, including routes such as /lesson/fly that never touch Clerk. Setting the header
 * keeps that decision on the server where it belongs.
 */
const MAX_HEADER_PATH = 512;

export default function proxy(request: NextRequest) {
  const response = NextResponse.next();
  // Bound the value: a header derived from a URL should not be arbitrarily long.
  const path =
    request.nextUrl.pathname.length > MAX_HEADER_PATH
      ? request.nextUrl.pathname.slice(0, MAX_HEADER_PATH)
      : request.nextUrl.pathname;
  response.headers.set("x-pathname", path);
  return response;
}

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
