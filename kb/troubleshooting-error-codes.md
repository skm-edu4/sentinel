# Troubleshooting Error Codes

Last updated: 2026-10-01 | Owner: Engineering Support

## Error 4032 — Payment Declined

**Meaning:** The issuing bank declined the card, or 3-D Secure (3DS) authentication did not complete.

**Fix steps to give the customer:**
1. Check the card number, expiry date, and CVV for typos.
2. Retry checkout — transient declines clear on the second attempt in most cases.
3. If it keeps failing, the customer should contact their bank (some banks block online charges by default) or use a different payment method.
4. Error 4032 is never caused by the customer's account balance being visible to us; do not speculate about funds.

## Error 5001 — Dashboard Failed to Load

**Meaning:** The dashboard front end could not fetch its payload from our servers.

**Fix steps:**
1. Clear the browser cache and site data, then reload.
2. Try a private/incognito window to rule out extensions.
3. Check status.sentinel.example for a live incident.
4. If it persists beyond 30 minutes, escalate to engineering with the account ID and approximate error time.

## Error 401 After Token Refresh (API v2)

**Meaning:** The access token was rejected after a successful refresh. Common cause: **token reuse detection** — the old token was used once more after the new one was issued.

**Fix steps:**
1. Ensure the client stops using the old token the instant a refresh returns a new one.
2. Store tokens atomically; concurrent workers must not refresh in parallel (single-flight refresh).
3. Check for clock skew larger than 5 minutes on the customer's servers.
4. Still failing? Escalate to API support with the client_id and a redacted request ID.

## Upload Failures Over Size Limit

- Photos: maximum **5 MB** per file.
- Documents (PDF, CSV): maximum **25 MB** per file.
- Uploads over the limit fail silently by design of the current client — tell the customer the exact limit and suggest compressing images. This is a known UX issue being tracked.

## Browser Requirements

Supported: Chrome, Edge, Firefox, Safari — current and previous major versions. Cleartext (HTTP) logins are not supported; the customer must be on HTTPS.
