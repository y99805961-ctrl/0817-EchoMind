import { expect, test } from "@playwright/test";

async function send(page: import("@playwright/test").Page, text: string) {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByTestId("chat-composer").fill(text);
  await page.getByTestId("send-button").click();
  await expect(page.getByTestId("assistant-message").last()).toBeVisible({ timeout: 60_000 });
}

test.describe.serial("Live Provider Smoke", () => {
  test("LIVE-001 Greeting", async ({ page }) => {
    await send(page, "你好");
    await expect(page.getByTestId("assistant-message").last()).not.toBeEmpty();
    await expect(page.getByTestId("intent-badge")).toBeVisible();
    await expect(page.getByTestId("primary-agent-badge")).toBeVisible();
  });

  test("LIVE-002 Refund", async ({ page }) => {
    await send(page, "我刚买的商品想退款");
    await expect(page.getByTestId("assistant-message").last()).not.toBeEmpty();
    await expect(page.getByTestId("primary-agent-badge")).toContainText("Billing");
  });

  test("LIVE-003 RAG Knowledge", async ({ page }) => {
    await send(page, "订单显示已发货是什么意思？");
    await expect(page.getByTestId("assistant-message").last()).not.toBeEmpty();
    await expect(page.getByTestId("knowledge-used-badge")).toBeVisible();
  });
});
