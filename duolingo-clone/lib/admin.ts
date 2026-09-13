import { auth } from "@clerk/nextjs/server";

export const getIsAdmin = async () => {
  const { userId } = await auth();

  if (!userId) return false;

  const adminIds =
    process.env.CLERK_ADMIN_IDS?.split(",")
      .map((id) => id.trim())
      .filter(Boolean) ?? [];

  return adminIds.includes(userId);
};
