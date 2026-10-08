# Known Issues — October 2026

Last updated: 2026-10-06 | Owner: Engineering | Reviewed weekly

## Android App 3.2.0 Crashes on Launch (Critical)

- **Affects:** Android app version 3.2.0 only.
- **Symptom:** Crash loop immediately after the splash screen; unsaved drafts in the editor may be lost.
- **Status:** Fix shipped as **version 3.2.1, rolling out from October 8** on Google Play.
- **Guidance:** There is no workaround — advise customers not to uninstall the app (reinstalling can discard local drafts). Confirm the replacement version is rolling out and escalate to the mobile team if a customer reports data loss, since draft recovery requires engineering help.

## CSV Export Returns an Empty File With Accented Characters

- **Affects:** Web app exports where any row contains accented or non-ASCII characters (for example José or Zoé).
- **Symptom:** The CSV downloads but contains only headers.
- **Root cause:** Character-encoding error in the export worker (ticket ENG-2291).
- **Workaround:** Export as **XLSX instead of CSV**, or temporarily romanize the affected names.
- **Fix:** Targeted for web release **4.18.3 (October 10)**.

## Dashboard Error 5001 (Intermittent)

- **Symptom:** Dashboard renders Error 5001 after login.
- **Workaround:** Clear site data / cache and reload; see the troubleshooting error-code guide for full steps.
- **Status:** Root cause under investigation; a stale-cache purge fixes it for most customers.

## Dark Mode Toggle Resets (Resolved)

- **Symptom:** Dark mode reverting to light after closing the tab.
- **Fix:** Shipped in web release **4.18.2** on September 15. If a customer still sees it, have them clear site data once.

## Not a Known Issue

Before confirming a bug is known, search this list first. If the behavior is absent here — for example a pricing-page typo, or an unexpected report discrepancy — it is **not yet tracked**: collect reproduction steps, browser/app version, account ID, and timestamps, then escalate to engineering. Never invent a fix timeline for untracked issues.
