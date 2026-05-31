# Holy Shifter v107 — Distribution Guide

## One-time setup (on your Mac)

You need three things before running the build script:

### 1. Install your Developer ID certificates

Go to [developer.apple.com → Certificates](https://developer.apple.com/account/resources/certificates/list) and create **two** certificates if you haven't already:

- **Developer ID Application** — signs the plugin bundles
- **Developer ID Installer** — signs the .pkg

Download and double-click each to install into your Keychain. Verify they're there:

```bash
security find-identity -v -p codesigning
```

You should see lines containing `Developer ID Application` and `Developer ID Installer` with your team ID `DU92Z6L82F`.

### 2. Generate an app-specific password

Go to [appleid.apple.com](https://appleid.apple.com) → Sign-In and Security → App-Specific Passwords → Generate one named "notarytool".

### 3. Store notarization credentials

```bash
xcrun notarytool store-credentials "notary-profile" \
  --apple-id "your@email.com" \
  --team-id "DU92Z6L82F" \
  --password "the-app-specific-password-you-just-generated"
```

This saves the credentials securely in your Keychain so the script can notarize without prompting.

## Building a distributable package

```bash
cd frequency-shifter/plugin
./scripts/build_and_notarize.sh
```

The script will:
1. Build a Universal Binary (Apple Silicon + Intel)
2. Code sign the VST3 and AU with your Developer ID
3. Package into a signed `.pkg` installer
4. Submit to Apple for notarization and wait
5. Staple the ticket to the `.pkg`
6. Verify it passes Gatekeeper

Output: `dist/Holy Shifter v107.pkg`

## Sharing with testers

Send `Holy Shifter v107.pkg` however you like — email, Dropbox, Google Drive, WeTransfer, etc. Your testers just double-click to install. No Terminal commands, no security warnings.

The installer places the plugins in:
- **VST3:** `/Library/Audio/Plug-Ins/VST3/Holy Shifter v107.vst3`
- **AU:** `/Library/Audio/Plug-Ins/Components/Holy Shifter v107.component`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No Developer ID certificates found" | Download & install certs from developer.apple.com |
| "Notarization credentials not found" | Run the `store-credentials` command above |
| Notarization rejected | Run `xcrun notarytool log <submission-id> --keychain-profile notary-profile` to see why |
| Script says "signing identity" name doesn't match | Edit `APP_SIGNING_ID` and `INSTALLER_SIGNING_ID` in the script to match the exact names from `security find-identity -v` |

## Important: update the signing identity names

The script defaults to `"Developer ID Application: Conduit AI (DU92Z6L82F)"`. If the name on your Apple Developer certificate is different (e.g. your personal name), update these two lines near the top of `build_and_notarize.sh`:

```bash
APP_SIGNING_ID="Developer ID Application: YOUR NAME HERE (DU92Z6L82F)"
INSTALLER_SIGNING_ID="Developer ID Installer: YOUR NAME HERE (DU92Z6L82F)"
```

Run `security find-identity -v -p codesigning` to see the exact string to use.
