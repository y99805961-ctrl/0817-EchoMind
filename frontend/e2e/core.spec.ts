import { expect, test, type Page } from "@playwright/test";

async function freshChat(page: Page) {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByTestId("chat-page")).toBeVisible();
  await expect(page.getByTestId("health-indicator")).toContainText("Online");
}

async function send(page: Page, text: string) {
  const composer = page.getByTestId("chat-composer");
  await composer.fill(text);
  await composer.press("Enter");
  await expect(page.getByTestId("assistant-message").last()).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId("loading-bubble")).toHaveCount(0);
}

test.describe.serial("Core deterministic E2E", () => {
  test("E2E-001 App Boot", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (message) => { if (message.type() === "error") consoleErrors.push(message.text()); });
    page.on("pageerror", (error) => consoleErrors.push(error.message));
    await freshChat(page);
    await expect(page.getByText("EchoMind").first()).toBeVisible();
    await expect(page.getByTestId("chat-composer")).toBeVisible();
    expect(consoleErrors).toEqual([]);
  });

  test("E2E-002 New Conversation", async ({ page }) => {
    await freshChat(page);
    const before = await page.getByTestId("conversation-item").count();
    await page.getByTestId("new-chat-button").click();
    await expect(page.getByTestId("conversation-item")).toHaveCount(before + 1);
    await expect(page.getByTestId("chat-composer")).toHaveValue("");
    const persisted = await page.evaluate(() => JSON.parse(localStorage.getItem("echomind-conversations") ?? "{}"));
    expect(persisted.state.conversations.length).toBe(before + 1);
    expect(persisted.state.conversations[0].id).not.toBe(persisted.state.conversations[1].id);
  });

  test("E2E-003 Greeting", async ({ page }) => {
    await freshChat(page);
    await send(page, "你好");
    await expect(page.getByTestId("intent-badge")).toHaveText("greeting");
    await expect(page.getByTestId("primary-agent-badge")).toContainText("General");
    await expect(page.getByTestId("diagnostics-panel")).toBeVisible();
  });

  test("E2E-004 Refund", async ({ page }) => {
    await freshChat(page);
    await send(page, "我刚买的商品想退款");
    await expect(page.getByTestId("intent-badge")).toHaveText("refund");
    await expect(page.getByTestId("primary-agent-badge")).toContainText("Billing");
    await expect(page.getByTestId("assistant-message").last()).not.toBeEmpty();
  });

  test("E2E-005 Technical Login", async ({ page }) => {
    await freshChat(page);
    await send(page, "我一直登录不上，提示 401");
    await expect(page.getByTestId("intent-badge")).toHaveText("technical_login");
    await expect(page.getByTestId("primary-agent-badge")).toContainText("Technical");
  });

  test("E2E-006 Human Handoff", async ({ page }) => {
    await freshChat(page);
    await send(page, "我要转人工客服");
    await expect(page.getByTestId("intent-badge")).toHaveText("human_handoff");
    await expect(page.getByTestId("diagnostics-panel")).toContainText("Escalation");
  });

  test("E2E-007 RAG Knowledge", async ({ page }) => {
    await freshChat(page);
    await send(page, "订单显示已发货是什么意思？");
    await expect(page.getByTestId("knowledge-used-badge")).toContainText("Knowledge Used");
    await expect(page.getByTestId("assistant-message").last()).not.toBeEmpty();
  });

  test("E2E-008 No RAG Greeting", async ({ page }) => {
    await freshChat(page);
    await send(page, "你好呀");
    await expect(page.getByTestId("knowledge-used-badge")).toContainText("Direct Answer");
  });

  test("E2E-009 Compound Routing", async ({ page }) => {
    await freshChat(page);
    await send(page, "我登录不上，而且信用卡好像被重复扣款了");
    const diagnostics = page.getByTestId("diagnostics-panel");
    await expect(diagnostics).toContainText("Technical");
    await expect(diagnostics).toContainText("Billing");
    await expect(diagnostics.getByText("Supporting")).toBeVisible();
  });

  test("E2E-010 Multi-turn Conversation", async ({ page }) => {
    await freshChat(page);
    const payloads: Array<{ conv_id?: string; message?: string }> = [];
    page.on("request", (request) => {
      if (request.url().includes("/api/chat") && request.method() === "POST") {
        try { payloads.push(JSON.parse(request.postData() ?? "{}") as { conv_id?: string; message?: string }); } catch { /* assertion below catches missing requests */ }
      }
    });
    await send(page, "我的订单号是 EM20260817001");
    await send(page, "我刚才说的订单号是什么？");
    expect(payloads).toHaveLength(2);
    expect(payloads[0].conv_id).toBeTruthy();
    expect(payloads[0].conv_id).toBe(payloads[1].conv_id);
    await expect(page.getByTestId("user-message")).toHaveCount(2);
  });

  test("E2E-011 Conversation Switch", async ({ page }) => {
    await freshChat(page);
    await send(page, "Conversation A 退款问题");
    await page.getByTestId("new-chat-button").click();
    await send(page, "Conversation B 登录问题");
    const conversations = page.getByTestId("conversation-item");
    await conversations.filter({ hasText: "Conversation A" }).click();
    await expect(page.getByTestId("user-message")).toHaveCount(1);
    await expect(page.getByTestId("user-message").first()).toContainText("Conversation A");
    await conversations.filter({ hasText: "Conversation B" }).click();
    await expect(page.getByTestId("user-message")).toHaveCount(1);
    await expect(page.getByTestId("user-message").first()).toContainText("Conversation B");
  });

  test("E2E-012 Search Playground", async ({ page }) => {
    await page.goto("/knowledge");
    await expect(page.getByTestId("knowledge-page")).toBeVisible();
    await page.getByTestId("knowledge-search-input").fill("退款多久到账");
    await page.getByTestId("knowledge-search-button").click();
    await expect(page.getByTestId("search-results")).toBeVisible();
    await expect(page.getByTestId("search-results")).toContainText("Timing");
    await expect(page.getByTestId("search-results")).toContainText("Fallback");
  });

  test("E2E-013 Knowledge Stats", async ({ page }) => {
    await page.goto("/knowledge");
    const knowledge = page.getByTestId("knowledge-page");
    await expect(knowledge).toBeVisible();
    await expect(knowledge.getByText("Dense model")).toBeVisible();
    await expect(knowledge.getByText("Reranker", { exact: true })).toBeVisible();
    await expect(knowledge.getByText("Index Status")).toBeVisible();
  });

  test("E2E-014 Backend Failure UX", async ({ page }) => {
    await freshChat(page);
    const armed = await page.request.post("/api/__e2e/fail-chat");
    expect(armed.ok()).toBeTruthy();
    await send(page, "请帮我查询订单状态");
    await expect(page.getByTestId("user-message").last()).toContainText("请帮我查询订单状态");
    await expect(page.getByTestId("assistant-message").last()).toContainText("Echo 暂时没有响应");
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
  });
});
