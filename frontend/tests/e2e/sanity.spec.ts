import { test, expect } from '@playwright/test';

test('dashboard loads without errors', async ({ page }) => {
  // Go to the dashboard
  await page.goto('/');

  // Check title
  await expect(page).toHaveTitle(/NFTool V3/);

  // Check for critical error overlays
  const errorOverlay = page.locator('text=Application error: a client-side exception has occurred');
  await expect(errorOverlay).not.toBeVisible();

  const runtimeError = page.locator('text=Runtime Error');
  await expect(runtimeError).not.toBeVisible();

  // Check if the footer indicates running status (or idle)
  // The WS status indicator is in the footer.
  // We look for the text "CONNECTED" or "CONNECTING".
  // Ideally, it should eventually be "CONNECTED".
  
  await expect(page.getByText('CONNECTED', { exact: true })).toBeVisible({ timeout: 10000 });

  // Check if datasets are loaded (dropdown)
  // The inspector has a select for "Predictors (X)".
  // We can check if it has options.
  const predictorSelect = page.locator('select').first();
  await expect(predictorSelect).toBeVisible();
  const optionCount = await predictorSelect.locator('option').count();
  expect(optionCount).toBeGreaterThan(0);
});
