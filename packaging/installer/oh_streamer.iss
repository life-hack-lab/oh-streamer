; Oh Streamer installer (Inno Setup 6). Built by packaging/build_installer.py, which passes:
;   /DAppName=...  /DAppVersion=...  /DStageDir=...  /DOutDir=...  /DOutputBaseName=...
; Keep this file UTF-8 with BOM (Inno reads the Chinese text correctly only then).
; Uninstall asks (three checkboxes, default keep) whether to delete: 1) GPU environments + uv cache + current.json,
; 2) downloaded subtitle models, 3) user data. /VERYSILENT keeps all unless /PURGE=envs,models,userdata (or all) is given.
; Test-only: /USERDATADIR=<folder named oh_streamer> replaces %LOCALAPPDATA%\oh_streamer (tests never touch the real folder).
; Test-only: /FORCEWEBVIEW2 runs the bundled WebView2 bootstrapper even when the runtime is already installed
; (normally it runs only when the runtime is missing). It changes the real WebView2 of the machine: use a sandbox.

#ifndef AppName
  #error AppName is not defined (pass /DAppName=...)
#endif
#ifndef AppVersion
  #error AppVersion is not defined
#endif
#ifndef StageDir
  #error StageDir is not defined
#endif
#ifndef OutDir
  #define OutDir "..\..\build-installer\out"
#endif
#ifndef OutputBaseName
  #define OutputBaseName "OhStreamer-Setup-" + AppVersion
#endif

; Must equal APP_MUTEX_NAME in src/oh_streamer/runtime_env/app_running.py (tests/test_build_installer_inno.py checks it).
#define AppMutexName "oh_streamer-app"

#define AppIcon "app\src\oh_streamer\resources\icons\app.ico"

[Setup]
; AppId never changes (even if the product is renamed): overwrite install and uninstall find the old copy by it.
AppId={{759A8B7A-6BC3-4A79-BA93-D39701E922CA}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=life-hack-lab
AppPublisherURL=https://github.com/life-hack-lab/oh-streamer
AppSupportURL=https://github.com/life-hack-lab/oh-streamer/issues
AppUpdatesURL=https://github.com/life-hack-lab/oh-streamer/releases
; No LicenseFile on purpose: no licence page in the wizard. LICENSE and THIRD_PARTY_NOTICES.md are installed in {app}.
AppVerName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName} Setup
; Per-user install, never asks for administrator rights.
PrivilegesRequired=lowest
DefaultDirName={userpf}\oh_streamer
DisableDirPage=no
UsePreviousAppDir=yes
DisableProgramGroupPage=yes
DisableReadyPage=no
; Peak disk use is during the first launch (downloads the GPU environment, about 9 GB).
ExtraDiskSpaceRequired=9663676416
OutputDir={#OutDir}
OutputBaseFilename={#OutputBaseName}
SetupIconFile={#StageDir}\{#AppIcon}
UninstallDisplayIcon={app}\{#AppIcon}
UninstallDisplayName={#AppName}
; Running program: every process that can lock the install folder holds this mutex, so Setup and the uninstaller
; stop with a "please close it" prompt. CloseApplications (Restart Manager) is off on purpose: it would force-end a
; pythonw process (a running background job) without the user knowing.
AppMutex={#AppMutexName}
CloseApplications=no
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/ultra64
SolidCompression=yes
ShowLanguageDialog=auto
LZMAUseSeparateProcess=yes
; KB. Must exceed 228 MB so ffprobe (same code as ffmpeg) is deduplicated against ffmpeg in the solid stream.
LZMADictionarySize=262144
LZMANumFastBytes=273

[Languages]
; Traditional Chinese first: it is the fallback when the Windows UI language matches neither (ShowLanguageDialog=auto).
; Languages\ChineseTraditional.isl is the unofficial translation pinned at tag is-6_7_3 of jrsoftware/issrc.
#if FileExists(SourcePath + "Languages\ChineseTraditional.isl")
Name: "chinesetraditional"; MessagesFile: "Languages\ChineseTraditional.isl"
#endif
Name: "english"; MessagesFile: "compiler:Default.isl"

#if FileExists(SourcePath + "Languages\ChineseTraditional.isl")
[LangOptions]
; Name the CJK font: the default Segoe UI relies on font linking, which Windows Sandbox lacks (Chinese drew as boxes).
chinesetraditional.DialogFontName=Microsoft JhengHei UI
#endif

[Messages]
SetupAppRunningError=偵測到 %1 正在執行。%n%n請先關閉它（包含右下角系統匣的圖示與背景處理中的工作），再按「確定」繼續；按「取消」則結束安裝。
UninstallAppRunningError=偵測到 %1 正在執行。%n%n請先關閉它（包含右下角系統匣的圖示與背景處理中的工作），再按「確定」繼續；按「取消」則結束解除安裝。
#if FileExists(SourcePath + "Languages\ChineseTraditional.isl")
; Added in Inno 6.5+ and missing from the 6.1.0-based translation (the other missing ones are download/extraction
; messages this installer never shows).
chinesetraditional.RetryCancelSelectAction=選取動作
chinesetraditional.RetryCancelRetry=再試一次 (&T)
chinesetraditional.RetryCancelCancel=取消
#endif

[Tasks]
Name: "desktopicon"; Description: "建立桌面捷徑 (&D)"; GroupDescription: "額外選項："

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Excludes: "redist\*"; Flags: recursesubdirs createallsubdirs ignoreversion
; Microsoft's WebView2 Evergreen bootstrapper: embedded in the installer, extracted to {tmp} only when needed, never put into {app}.
Source: "{#StageDir}\redist\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: dontcopy

[InstallDelete]
; Overwrite install: remove the old program and Python so files deleted in the new version do not linger.
; NEVER list envs, uv-cache, logs or current.json here: they are created after install and must survive updates.
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\licenses"

[UninstallDelete]
; Only installer-owned folders and logs under {app}; envs, uv-cache and current.json are handled in [Code] (the user's choice).
Type: filesandordirs; Name: "{app}\app"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\licenses"
Type: filesandordirs; Name: "{app}\logs"
Type: dirifempty; Name: "{app}"

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\launch.pyw"""; WorkingDir: "{app}\app"; IconFilename: "{app}\{#AppIcon}"; AppUserModelID: "oh_streamer.Desktop"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\launch.pyw"""; WorkingDir: "{app}\app"; IconFilename: "{app}\{#AppIcon}"; AppUserModelID: "oh_streamer.Desktop"; Tasks: desktopicon
; Opens setup mode (reinstall, switch version, export diagnostics) even when the install is ready.
Name: "{autoprograms}\{#AppName} 修復與診斷"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\launch.pyw"" --setup"; WorkingDir: "{app}\app"; IconFilename: "{app}\{#AppIcon}"

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\app\launch.pyw"""; WorkingDir: "{app}\app"; Description: "啟動 {#AppName}"; Flags: nowait postinstall skipifsilent

[Code]
const
  MaxInstallRootLength = 110;  { deepest installed path is about 138 characters longer than the root; Windows limit is 260 }

function NormalizeDir(const S: String): String;
begin
  Result := RemoveBackslashUnlessRoot(Trim(S));
end;

function IsSameOrUnder(const Path, Base: String): Boolean;
var
  P, B: String;
begin
  Result := False;
  B := Lowercase(NormalizeDir(Base));
  if B = '' then Exit;
  P := Lowercase(NormalizeDir(Path));
  Result := (P = B) or (Pos(AddBackslash(B), AddBackslash(P)) = 1);
end;

function IsUnderConst(const Path, ConstStr: String): Boolean;
var
  Base: String;
begin
  Result := False;
  try
    Base := ExpandConstant(ConstStr);
    Result := IsSameOrUnder(Path, Base);
  except
  end;
end;

function IsAdminOnlyLocation(const Path: String): Boolean;
begin
  Result := IsUnderConst(Path, '{commonpf}') or IsUnderConst(Path, '{commonpf32}')
         or IsUnderConst(Path, '{commonpf64}') or IsUnderConst(Path, '{win}')
         or IsUnderConst(Path, '{sys}');
end;

{ Nearest folder that already exists (the folder itself, else the closest parent); '' if none (e.g. missing drive). }
function NearestExistingDir(const Path: String): String;
var
  D, Parent: String;
begin
  Result := '';
  D := NormalizeDir(Path);
  while D <> '' do
  begin
    if DirExists(D) then
    begin
      Result := D;
      Exit;
    end;
    Parent := ExtractFileDir(D);
    if (Parent = D) or (Parent = '') then Exit;
    D := Parent;
  end;
end;

function CanWriteIn(const Dir: String): Boolean;
var
  Probe: String;
begin
  Probe := AddBackslash(Dir) + 'oh_streamer_write_test_' + IntToStr(Random(1000000000)) + '.tmp';
  Result := SaveStringToFile(Probe, 'x', False);
  if Result then DeleteFile(Probe);
end;

{ Shows the reason; in a silent install nobody can pick another folder, so end setup instead of looping on the page. }
procedure Reject(const Msg: String);
begin
  if WizardSilent then
  begin
    Log('Install location rejected: ' + Msg);
    Abort;
  end;
  MsgBox(Msg, mbError, MB_OK);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir, Existing: String;
begin
  Result := True;
  if CurPageID <> wpSelectDir then Exit;
  Dir := NormalizeDir(WizardDirValue);

  if Length(Dir) > MaxInstallRootLength then
  begin
    Reject('這個安裝位置的路徑太長了（目前 ' + IntToStr(Length(Dir)) + ' 個字元，上限 ' + IntToStr(MaxInstallRootLength) + ' 個）。' + #13#10#13#10 +
      'Windows 的路徑長度上限是 260 個字元，而程式安裝後最深的檔案路徑會比安裝資料夾再多出約 140 個字元，太長會導致程式或 GPU 環境無法正常安裝。' + #13#10#13#10 +
      '請改選較短的位置，例如 D:\oh_streamer。');
    Result := False;
    Exit;
  end;

  if IsAdminOnlyLocation(Dir) then
  begin
    Reject('這個位置（Program Files 或 Windows 資料夾底下）需要系統管理員權限才能寫入，而本程式只為目前的使用者安裝，不需要也不會要求系統管理員權限。' + #13#10#13#10 +
      '請改選其他位置，例如預設位置，或其他磁碟機上的資料夾（如 D:\oh_streamer）。');
    Result := False;
    Exit;
  end;

  Existing := NearestExistingDir(Dir);
  if (Existing = '') or (not CanWriteIn(Existing)) then
  begin
    Reject('無法在這個位置建立檔案：磁碟機不存在，或目前的使用者沒有寫入權限（可能需要系統管理員權限）。' + #13#10#13#10 +
      '請改選其他位置，例如預設位置，或其他磁碟機上你自己的資料夾。');
    Result := False;
    Exit;
  end;
end;


{ ---- Uninstall: what to delete besides the program ---- }

const
  AttrDirectory = $10;
  AttrReparsePoint = $400;
  UserDirLeaf = 'oh_streamer';
  ModelsLeaf = 'subtitle_models';
  MaxTreeDepth = 60;

var
  PurgeEnvs, PurgeModels, PurgeUserData: Boolean;
  UserDataDirOverride: String;

function SizeText(Bytes: Int64): String;
var
  MB, G: Int64;
begin
  MB := Bytes div 1048576;
  if MB >= 1024 then
  begin
    G := MB div 1024;
    Result := IntToStr(G) + '.' + IntToStr(((MB mod 1024) * 10) div 1024) + ' GB';
  end
  else if MB >= 1 then
    Result := IntToStr(MB) + ' MB'
  else
    Result := '不到 1 MB';
end;

function DirSize(const Dir: String; Depth: Integer): Int64;
var
  FindRec: TFindRec;
  P: String;
begin
  Result := 0;
  if Depth > MaxTreeDepth then Exit;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  try
    repeat
      if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
      begin
        P := AddBackslash(Dir) + FindRec.Name;
        if (FindRec.Attributes and AttrReparsePoint) <> 0 then
          { a link: not counted and never followed }
        else if (FindRec.Attributes and AttrDirectory) <> 0 then
          Result := Result + DirSize(P, Depth + 1)
        else
          Result := Result + Int64(FindRec.SizeHigh) * 4294967296 + Int64(FindRec.SizeLow);
      end;
    until not FindNext(FindRec);
  finally
    FindClose(FindRec);
  end;
end;

function LastPart(const Path: String): String;
begin
  Result := ExtractFileName(RemoveBackslashUnlessRoot(Path));
end;

{ Only ever delete below a folder we named ourselves: not empty, not a drive root, and the leaf name is what we expect ('' = any). }
function SafeBase(const Dir, ExpectedLeaf: String): Boolean;
var
  D: String;
begin
  D := RemoveBackslashUnlessRoot(Trim(Dir));
  Result := (Length(D) > 3) and (Pos(':', D) = 2) and (ExtractFileDir(D) <> D);
  if Result and (ExpectedLeaf <> '') then
    Result := CompareText(LastPart(D), ExpectedLeaf) = 0;
end;

{ Deletes a folder's contents (links are removed, never followed) and then the folder; failures go to the log and never stop the uninstall. }
procedure PurgeTree(const Dir: String; Depth: Integer);
var
  FindRec: TFindRec;
  P: String;
begin
  if Depth > MaxTreeDepth then
  begin
    Log('Purge: too deep, left in place: ' + Dir);
    Exit;
  end;
  if FindFirst(AddBackslash(Dir) + '*', FindRec) then
  try
    repeat
      if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
      begin
        P := AddBackslash(Dir) + FindRec.Name;
        if (FindRec.Attributes and AttrReparsePoint) <> 0 then
        begin
          if not (DeleteFile(P) or RemoveDir(P)) then Log('Purge: could not remove link ' + P);
        end
        else if (FindRec.Attributes and AttrDirectory) <> 0 then
          PurgeTree(P, Depth + 1)
        else if not DeleteFile(P) then
        begin
          if not DelTree(P, False, True, False) then Log('Purge: could not delete file ' + P);
        end;
      end;
    until not FindNext(FindRec);
  finally
    FindClose(FindRec);
  end;
  if DirExists(Dir) and not RemoveDir(Dir) then Log('Purge: could not remove folder ' + Dir);
end;

procedure PurgeChild(const Base, BaseLeaf, Child: String);
var
  Target: String;
begin
  if not SafeBase(Base, BaseLeaf) then
  begin
    Log('Purge: refused, unsafe base folder "' + Base + '"');
    Exit;
  end;
  Target := AddBackslash(RemoveBackslashUnlessRoot(Base)) + Child;
  if (Child = '') or not IsSameOrUnder(Target, Base) or (CompareText(NormalizeDir(Target), NormalizeDir(Base)) = 0) then
  begin
    Log('Purge: refused, ' + Target + ' is not below ' + Base);
    Exit;
  end;
  if DirExists(Target) then
  begin
    Log('Purge: deleting folder ' + Target);
    PurgeTree(Target, 0);
  end
  else if FileExists(Target) then
  begin
    Log('Purge: deleting file ' + Target);
    if not DeleteFile(Target) then Log('Purge: could not delete file ' + Target);
  end
  else
    Log('Purge: nothing at ' + Target);
end;

{ Every entry of the user data folder except the one named Keep ('' = keep nothing). }
procedure PurgeUserDataExcept(const Base, Keep: String);
var
  FindRec: TFindRec;
  Names: TStringList;
  I: Integer;
begin
  if not SafeBase(Base, UserDirLeaf) then
  begin
    Log('Purge: refused, unsafe user data folder "' + Base + '"');
    Exit;
  end;
  Names := TStringList.Create;
  try
    if FindFirst(AddBackslash(Base) + '*', FindRec) then
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') and
           ((Keep = '') or (CompareText(FindRec.Name, Keep) <> 0)) then
          Names.Add(FindRec.Name);
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
    for I := 0 to Names.Count - 1 do
      PurgeChild(Base, UserDirLeaf, Names[I]);
  finally
    Names.Free;
  end;
end;

function UserDataDir: String;
begin
  if UserDataDirOverride <> '' then
    Result := UserDataDirOverride
  else
    Result := ExpandConstant('{localappdata}\' + UserDirLeaf);
end;

function HasToken(const List, Token: String): Boolean;
var
  L: String;
begin
  L := Lowercase(List);
  StringChangeEx(L, ' ', '', True);
  L := ',' + L + ',';
  Result := (Pos(',' + Token + ',', L) > 0) or (Pos(',all,', L) > 0);
end;

function AskWhatToDelete: Boolean;
var
  Form: TSetupForm;
  Intro, Desc1, Desc2, Desc3: TNewStaticText;
  Cb1, Cb2, Cb3: TNewCheckBox;
  BtnOk, BtnCancel: TNewButton;
  EnvsDir, CacheDir, ModelsDir, UserDir: String;
  SzEnvs, SzModels, SzUser: Int64;
  Y: Integer;
begin
  EnvsDir := ExpandConstant('{app}\envs');
  CacheDir := ExpandConstant('{app}\uv-cache');
  UserDir := UserDataDir;
  ModelsDir := AddBackslash(UserDir) + ModelsLeaf;
  SzEnvs := DirSize(EnvsDir, 0) + DirSize(CacheDir, 0);
  SzModels := DirSize(ModelsDir, 0);
  SzUser := DirSize(UserDir, 0) - SzModels;
  if SzUser < 0 then SzUser := 0;

  Form := CreateCustomForm(520, 340, False, True);
  try
    Form.Caption := '解除安裝 {#AppName}';
    Form.Position := poScreenCenter;

    Intro := TNewStaticText.Create(Form);
    Intro.Parent := Form;
    Intro.AutoSize := False;
    Intro.WordWrap := True;
    Intro.Left := ScaleX(16);
    Intro.Top := ScaleY(14);
    Intro.Width := Form.ClientWidth - ScaleX(32);
    Intro.Height := ScaleY(48);
    Intro.Caption := '即將移除 {#AppName} 程式本身。下面三項預設都會「保留」，之後重新安裝就不必再下載或重設；勾選的項目才會一併刪除：';

    Y := ScaleY(70);

    Cb1 := TNewCheckBox.Create(Form);
    Cb1.Parent := Form;
    Cb1.Left := ScaleX(16);
    Cb1.Top := Y;
    Cb1.Width := Form.ClientWidth - ScaleX(32);
    Cb1.Height := ScaleY(20);
    if DirExists(EnvsDir) or DirExists(CacheDir) then
      Cb1.Caption := 'GPU 執行環境與下載快取（' + SizeText(SzEnvs) + '）'
    else
      Cb1.Caption := 'GPU 執行環境與下載快取（約 5 GB，目前沒有）';
    Cb1.Checked := PurgeEnvs;
    Desc1 := TNewStaticText.Create(Form);
    Desc1.Parent := Form;
    Desc1.AutoSize := False;
    Desc1.WordWrap := True;
    Desc1.Left := ScaleX(36);
    Desc1.Top := Y + ScaleY(22);
    Desc1.Width := Form.ClientWidth - ScaleX(52);
    Desc1.Height := ScaleY(34);
    Desc1.Caption := '第一次啟動時下載的運算環境。刪除後，重新安裝要再花時間下載一次。';

    Y := Y + ScaleY(72);

    Cb2 := TNewCheckBox.Create(Form);
    Cb2.Parent := Form;
    Cb2.Left := ScaleX(16);
    Cb2.Top := Y;
    Cb2.Width := Form.ClientWidth - ScaleX(32);
    Cb2.Height := ScaleY(20);
    if DirExists(ModelsDir) then
      Cb2.Caption := '已下載的字幕模型（' + SizeText(SzModels) + '）'
    else
      Cb2.Caption := '已下載的字幕模型（目前沒有，通常數 GB）';
    Cb2.Checked := PurgeModels;
    Desc2 := TNewStaticText.Create(Form);
    Desc2.Parent := Form;
    Desc2.AutoSize := False;
    Desc2.WordWrap := True;
    Desc2.Left := ScaleX(36);
    Desc2.Top := Y + ScaleY(22);
    Desc2.Width := Form.ClientWidth - ScaleX(52);
    Desc2.Height := ScaleY(34);
    Desc2.Caption := '本機字幕用的語音與翻譯模型。刪除後，要用本機字幕得重新下載。';

    Y := Y + ScaleY(72);

    Cb3 := TNewCheckBox.Create(Form);
    Cb3.Parent := Form;
    Cb3.Left := ScaleX(16);
    Cb3.Top := Y;
    Cb3.Width := Form.ClientWidth - ScaleX(32);
    Cb3.Height := ScaleY(20);
    if DirExists(UserDir) then
      Cb3.Caption := '使用者資料：下載佇列與各項設定（' + SizeText(SzUser) + '）'
    else
      Cb3.Caption := '使用者資料：下載佇列與各項設定（目前沒有）';
    Cb3.Checked := PurgeUserData;
    Desc3 := TNewStaticText.Create(Form);
    Desc3.Parent := Form;
    Desc3.AutoSize := False;
    Desc3.WordWrap := True;
    Desc3.Left := ScaleX(36);
    Desc3.Top := Y + ScaleY(22);
    Desc3.Width := Form.ClientWidth - ScaleX(52);
    Desc3.Height := ScaleY(50);
    Desc3.Caption := '佇列、播放與其他設定。你自己選的輸出影片資料夾不會被刪除。只勾這一項時，已下載的字幕模型仍會保留。';

    BtnOk := TNewButton.Create(Form);
    BtnOk.Parent := Form;
    BtnOk.Caption := '解除安裝';
    BtnOk.ModalResult := mrOk;
    BtnOk.Width := ScaleX(100);
    BtnOk.Height := ScaleY(26);
    BtnOk.Left := Form.ClientWidth - ScaleX(220);
    BtnOk.Top := Form.ClientHeight - ScaleY(42);
    BtnOk.Default := True;

    BtnCancel := TNewButton.Create(Form);
    BtnCancel.Parent := Form;
    BtnCancel.Caption := '取消';
    BtnCancel.ModalResult := mrCancel;
    BtnCancel.Cancel := True;
    BtnCancel.Width := ScaleX(100);
    BtnCancel.Height := ScaleY(26);
    BtnCancel.Left := Form.ClientWidth - ScaleX(112);
    BtnCancel.Top := Form.ClientHeight - ScaleY(42);

    Result := Form.ShowModal = mrOk;
    if Result then
    begin
      PurgeEnvs := Cb1.Checked;
      PurgeModels := Cb2.Checked;
      PurgeUserData := Cb3.Checked;
    end;
  finally
    Form.Free;
  end;
end;

{ ---- WebView2 runtime (needed by the app window; Microsoft's documented detection is the "pv" value) ---- }
const
  WebView2Guid = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  WebView2Bootstrapper = 'MicrosoftEdgeWebview2Setup.exe';

function WebView2VersionOk(const V: String): Boolean;
begin
  Result := (Trim(V) <> '') and (Trim(V) <> '0.0.0.0');
end;

function WebView2Installed: Boolean;
var
  V: String;
begin
  Result := False;
  if RegQueryStringValue(HKLM64, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + WebView2Guid, 'pv', V) and WebView2VersionOk(V) then
  begin
    Log('WebView2 runtime found (HKLM WOW6432Node), version ' + V);
    Result := True;
  end
  else if RegQueryStringValue(HKLM64, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + WebView2Guid, 'pv', V) and WebView2VersionOk(V) then
  begin
    Log('WebView2 runtime found (HKLM), version ' + V);
    Result := True;
  end
  else if RegQueryStringValue(HKLM32, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + WebView2Guid, 'pv', V) and WebView2VersionOk(V) then
  begin
    Log('WebView2 runtime found (HKLM 32-bit view), version ' + V);
    Result := True;
  end
  else if RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\' + WebView2Guid, 'pv', V) and WebView2VersionOk(V) then
  begin
    Log('WebView2 runtime found (HKCU, per-user), version ' + V);
    Result := True;
  end;
end;

{ A bare switch such as /FORCEWEBVIEW2: the param constant only reads /NAME=value forms. }
function CmdLineHasSwitch(const Name: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), Name) = 0 then
    begin
      Result := True;
      Exit;
    end;
end;

procedure EnsureWebView2;
var
  Forced, Started: Boolean;
  Code: Integer;
  OldStatus: String;
begin
  Forced := CmdLineHasSwitch('/FORCEWEBVIEW2');
  if Forced then
    Log('WebView2: /FORCEWEBVIEW2 given (test only), running the bootstrapper even if the runtime is present');
  if (not Forced) and WebView2Installed then
  begin
    Log('WebView2: runtime already present, bootstrapper not run');
    Exit;
  end;
  if not Forced then
    Log('WebView2: runtime not found, running the bootstrapper');
  OldStatus := WizardForm.StatusLabel.Caption;
  WizardForm.StatusLabel.Caption := '正在安裝 Microsoft Edge WebView2 執行階段…';
  WizardForm.FilenameLabel.Caption := '';
  try
    ExtractTemporaryFile(WebView2Bootstrapper);
    Started := Exec(ExpandConstant('{tmp}\' + WebView2Bootstrapper), '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, Code);
    if not Started then
      Log('WebView2: could not start the bootstrapper, ' + SysErrorMessage(Code))
    else
      Log('WebView2: bootstrapper exit code ' + IntToStr(Code));
  except
    Started := False;
    Code := -1;
    Log('WebView2: bootstrapper failed with an exception: ' + GetExceptionMessage);
  end;
  WizardForm.StatusLabel.Caption := OldStatus;
  if Started and (Code = 0) and WebView2Installed then
    Log('WebView2: runtime installed')
  else
  begin
    Log('WebView2: runtime still not available after the bootstrapper (the app shows a download link when opened)');
    if not WizardSilent then
      MsgBox('無法安裝 Microsoft Edge WebView2 執行階段。{#AppName} 已安裝完成，但第一次開啟時會顯示 WebView2 的下載連結，安裝後即可使用。', mbInformation, MB_OK);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    RemoveDir(ExpandConstant('{app}\redist'));      { left empty by createallsubdirs although redist\* is excluded }
    EnsureWebView2;
  end;
end;

function InitializeUninstall(): Boolean;
var
  List: String;
begin
  Result := True;
  List := ExpandConstant('{param:PURGE|}');
  UserDataDirOverride := RemoveBackslashUnlessRoot(Trim(ExpandConstant('{param:USERDATADIR|}')));
  PurgeEnvs := HasToken(List, 'envs');
  PurgeModels := HasToken(List, 'models');
  PurgeUserData := HasToken(List, 'userdata');
  if UserDataDirOverride <> '' then
    Log('Uninstall: user data folder override (test only): ' + UserDataDirOverride);
  if not UninstallSilent then
    Result := AskWhatToDelete;
  if Result then
    Log('Uninstall choices: envs=' + IntToStr(Ord(PurgeEnvs)) + ' models=' + IntToStr(Ord(PurgeModels)) +
        ' userdata=' + IntToStr(Ord(PurgeUserData)) + ' (silent=' + IntToStr(Ord(UninstallSilent)) + ', /PURGE="' + List + '")')
  else
    Log('Uninstall cancelled by the user.');
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AppDir, UserDir: String;
begin
  if CurUninstallStep <> usPostUninstall then Exit;
  AppDir := ExpandConstant('{app}');
  UserDir := UserDataDir;
  { logs always go ([UninstallDelete] also lists them); the rest only on request }
  PurgeChild(AppDir, '', 'logs');
  if PurgeEnvs then
  begin
    PurgeChild(AppDir, '', 'envs');
    PurgeChild(AppDir, '', 'uv-cache');
    PurgeChild(AppDir, '', 'current.json');
  end
  else
    Log('Purge: kept GPU environments and cache');
  if PurgeModels then
    PurgeChild(UserDir, UserDirLeaf, ModelsLeaf)
  else
    Log('Purge: kept subtitle models');
  if PurgeUserData then
  begin
    if PurgeModels then PurgeUserDataExcept(UserDir, '') else PurgeUserDataExcept(UserDir, ModelsLeaf);
    if SafeBase(UserDir, UserDirLeaf) and DirExists(UserDir) then RemoveDir(UserDir);
  end
  else
    Log('Purge: kept user data');
  if SafeBase(AppDir, '') and DirExists(AppDir) then
    RemoveDir(AppDir);   { only succeeds when empty; Inno removes it again after exit once its own files are gone }
end;
