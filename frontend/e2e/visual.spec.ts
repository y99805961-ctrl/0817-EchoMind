import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const screenshotDir = resolve(process.cwd(), "..", "data/eval/results/e2e/screenshots");

test("Visual QA at 1440x900", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByTestId("chat-page")).toBeVisible();
  await page.screenshot({ path: resolve(screenshotDir, "chat-empty.png"), fullPage: true });

  await page.getByTestId("chat-composer").fill("订单显示已发货是什么意思？");
  await page.getByTestId("send-button").click();
  await expect(page.getByTestId("assistant-message").last()).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("diagnostics-panel")).toBeVisible();
  await page.screenshot({ path: resolve(screenshotDir, "chat-answered-diagnostics.png"), fullPage: true });

  await page.goto("/knowledge");
  await expect(page.getByTestId("knowledge-page")).toBeVisible();
  await page.screenshot({ path: resolve(screenshotDir, "knowledge.png"), fullPage: true });
});
