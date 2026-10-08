# Password & Account Recovery

Last updated: 2026-09-20 | Owner: Identity Team

## Password Reset

Self-service reset path: **Sign in page → Forgot password → enter account email**.

- The reset email arrives from `no-reply@sentinel.example` within **5 minutes**.
- If it has not arrived after 15 minutes, the customer should check spam, promotions, and quarantine folders, then whitelist the sender and request again.
- Reset links expire after **60 minutes** and can be used once.
- If the email still does not arrive after two attempts, the customer should confirm which email address the account is registered to (support can confirm the masked address, e.g. `j****@company.com`, after identity verification).

## Account Lockout

- Five consecutive failed sign-in attempts trigger a **30-minute lockout**.
- The lock clears automatically after 30 minutes — no ticket needed.
- A human agent may unlock sooner, but only after verifying identity (order number on file plus access to the account email).

## Two-Factor Authentication (2FA)

- Supported factors: **authenticator app (TOTP)** and **SMS**.
- Setup path: Settings → Security → Two-factor authentication.
- Losing the 2FA device requires identity verification to regain access — escalate to the identity queue.

## Changing the Account Email Address

Email changes are **not self-serve**. The account owner must contact support and complete identity verification (proof of access to the old address plus an order number). Once verified, the change is applied within one business day.

## Merging Accounts

Two accounts with different email addresses can be merged only after both addresses are verified. This is always a human-agent operation — gather both account IDs and escalate to the identity queue. Do not promise a timeline beyond one business day.
