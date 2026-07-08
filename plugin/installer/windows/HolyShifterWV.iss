; Inno Setup script — Holy Shifter WV (WebView UI) Windows VST3 installer.
; Built in CI on windows-latest. Version + payload path are injected via ISCC /D defines:
;   ISCC.exe /DAppVersion=0.2.5 /DPayloadDir="<abs path to ...\Release\VST3>" HolyShifterWV.iss
;
; The build produces the plug-in as a bundle FOLDER:
;   <PayloadDir>\Holy Shifter WV.vst3\Contents\x86_64-win\Holy Shifter WV.vst3
; so the whole folder is installed into the shared VST3 directory
;   C:\Program Files\Common Files\VST3  ({commoncf64}\VST3)

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\..\build-wv\FrequencyShifter_artefacts\Release\VST3"
#endif

#define AppName "Holy Shifter WV"
#define AppPublisher "Heathen Machines"
#define Vst3Name "Holy Shifter WV.vst3"

[Setup]
; Stable AppId so upgrades replace the previous install instead of stacking.
AppId={{7B3D9E14-6C2A-4F58-A9D1-2E8C7F0A5B34}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={commoncf64}\VST3
DisableDirPage=yes
DisableProgramGroupPage=yes
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#Vst3Name}\Contents\x86_64-win\{#Vst3Name}
OutputDir=Output
OutputBaseFilename=HolyShifterWV-{#AppVersion}-Windows-Setup
Compression=lzma2
SolidCompression=yes
; Force the native 64-bit Common Files path (avoids WoW64 redirection).
ArchitecturesInstallIn64BitMode=x64
; Writing to Program Files\Common Files needs elevation.
PrivilegesRequired=admin
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Install the entire .vst3 bundle folder recursively; the .exp/.lib link
; byproducts sitting next to it in the build dir are intentionally not matched.
Source: "{#PayloadDir}\{#Vst3Name}\*"; DestDir: "{app}\{#Vst3Name}"; \
  Flags: recursesubdirs createallsubdirs ignoreversion

[UninstallDelete]
Type: filesandordirs; Name: "{app}\{#Vst3Name}"
