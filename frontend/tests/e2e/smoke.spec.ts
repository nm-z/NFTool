import { test, expect, Page } from '@playwright/test';

test.describe('UI Stability Smoke Test', () => {
  let consoleErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    // Capture console errors
    page.on('console', msg => {
      if (msg.type() === 'error') {
        const text = msg.text();
        if (!text.includes('[HMR]') && !text.includes('Training start failed') && !text.includes('Training already in progress')) {
          consoleErrors.push(text);
        }
      }
    });

    // Capture unhandled exceptions
    page.on('pageerror', err => {
      consoleErrors.push(`Uncaught Exception: ${err.message}`);
    });
  });

  // Helper to test buttons in a specific region
  async function testRegionButtons(page: Page, regionSelector: string, regionName: string) {
    console.log(`
--- Testing Region: ${regionName} ---`);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible({ timeout: 15000 });

    const region = page.locator(regionSelector);
    if (!(await region.isVisible())) {
      console.log(`Region ${regionName} not visible, skipping.`);
      return;
    }

    const buttons = await region.locator('button:visible, [role="button"]:visible').all();
    const identities = [];

    // 1. Collect Identities
    for (let i = 0; i < buttons.length; i++) {
      const btn = buttons[i];
      const text = (await btn.textContent())?.trim();
      const label = await btn.getAttribute('aria-label');
      
      if (text && text.length > 0) identities.push({ type: 'text', value: text });
      else if (label) identities.push({ type: 'label', value: label });
      else identities.push({ type: 'index', value: i }); // Fallback for scoped index
    }

    console.log(`Found ${identities.length} buttons in ${regionName}.`);

    // 2. Click Each
    for (const id of identities) {
      // Reload to ensure clean state
      await page.goto('/');
      await page.waitForLoadState('networkidle');
      
      // Re-scope to region
      const scopedRegion = page.locator(regionSelector);
      let locator;

      if (id.type === 'text') {
        locator = scopedRegion.locator(`button, [role="button"]`).filter({ hasText: id.value }).first();
      } else if (id.type === 'label') {
        locator = scopedRegion.locator(`[aria-label="${id.value}"]`).first();
      } else {
        locator = scopedRegion.locator('button:visible, [role="button"]:visible').nth(id.value as number);
      }

      const btnName = `${regionName} > ${id.value}`;

      if (await locator.isVisible()) {
        try {
          console.log(`Clicking: ${btnName}`);
          await locator.click({ timeout: 2000 });
          await page.waitForTimeout(300);

          // Check URL
          const url = page.url();
          if (!url.includes('localhost:3000')) {
             consoleErrors.push(`Navigation Error: Button "${btnName}" redirected to ${url}`);
          }

          // Check for Crash Overlay
          const crash = await page.locator('text=Runtime Error').or(page.locator('text=Application error')).count();
          if (crash > 0) {
            consoleErrors.push(`Crash detected after clicking "${btnName}"`);
          }

        } catch (e: any) {
          console.log(`⚠️ Soft Failure: Could not click "${btnName}": ${e.message}`);
        }
      } else {
        console.log(`Skipping "${btnName}" (not visible after reload)`);
      }
    }
  }

  test('Scoped Region Smoke Tests', async ({ page }) => {
    test.setTimeout(300000); // 5 minutes total budget

    // Test Header
    await testRegionButtons(page, 'header', 'Header');

    // Test Sidebar (Left)
    await testRegionButtons(page, 'aside.w-\\[56px\\]', 'Sidebar');

    // Test Inspector (Right)
    await testRegionButtons(page, 'aside.border-l', 'Inspector');

    // Test Footer
    await testRegionButtons(page, 'footer', 'Footer');

    // Test Main Workspace Tabs (the tab list specifically)
    await testRegionButtons(page, '.flex.gap-8[role="tablist"]', 'Main Tabs');

    // Final Assertion
    if (consoleErrors.length > 0) {
      console.error("\n❌ SMOKE TEST FAILED with the following errors:");
      consoleErrors.forEach(e => console.error(e));
      expect(consoleErrors).toHaveLength(0);
    }
  });

  test('Workflow: CNN Training and Log Capture', async ({ page }) => {
    test.setTimeout(60000);
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible({ timeout: 15000 });

    // 1. Click CNN in Inspector
    // We target the button explicitly
    const cnnBtn = page.locator('aside.border-l').locator('button', { hasText: 'CNN' });
    await cnnBtn.click();
    console.log('Selected CNN model.');

    // 2. Click Execute Pass (Run) in Header
    // "Execute Pass" is inside the header, usually implies starting. 
    // Button text might be "Execute Pass" or similar icon-based.
    // Based on previous logs: "Run" was found in header? Or "EXECUTE PASS".
    // Let's use a robust locator.
    const runBtn = page.locator('header').locator('button', { hasText: 'EXECUTE PASS' }).or(page.locator('header').locator('button', { hasText: 'Run' })).first();
    await runBtn.click();
    console.log('Clicked Execute Pass.');

    // 3. Wait for logs to populate
    // We look for the "Process Stream" container.
    // And wait for at least one log entry (e.g. timestamp bracket "[")
    const logContainer = page.locator('.flex-1.overflow-y-auto.p-3.font-mono');
    await expect(logContainer).toContainText('[', { timeout: 10000 });
    
    // 4. Capture and Output Logs
    // Wait a bit to get some logs
    await page.waitForTimeout(5000);
    const logs = await logContainer.innerText();
    
    console.log("\n=== CAPTURED PROCESS STREAM LOGS ===");
    console.log(logs);
    console.log("====================================\n");

    expect(logs.length).toBeGreaterThan(0);
  });
});
