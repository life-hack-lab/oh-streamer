-- oh_seek.lua: seeking + seekbar for oh_player.py's seekable player (run mpv with --osc=no).
--
-- Seeking: targets inside mpv's demuxer cache are seeked natively; anything else is published as
-- user-data/oh/seek-request ("<counter>:<seconds>") for the Python controller, which restarts Lada there
-- and loads the new segment. Positions are absolute source time (mpv runs with --rebase-start-time=no).
-- The user's pause state is remembered with the first request of a chain and a "處理中…" state (orange playhead and
-- label at the target) is shown until the new segment plays; then the user's pause state is restored. A seek does not
-- pause playback here: while the user is not paused, Python keeps the old timeline playing (seek_tail.py: the restored
-- video mpv has cached, then unrestored "passthrough" frames in the same stream; user-data/oh/tail = "1" meanwhile)
-- and appends the new segment behind it once ready, so mpv moves on by itself; otherwise Python pauses playback
-- itself and replaces the segment when ready. Quality / settings requests always pause at once. Python's segment loads
-- (replace or append) are matched by user-data/oh/served (the request being loaded) on file-loaded.
-- While Python works on a request, it publishes user-data/oh/progress (JSON {req = the request being served
-- ("init…" for a source's first segment), pct, stage, eta?}, "" when done): the orange busy text in the control bar
-- then shows the percentage, the stage and the remaining time (the picture itself stays free; only an initial load
-- with nothing on screen yet gets a centred progress box). `script-message oh-seek <seconds>` seeks like a key press
-- (cache check included; tests drive the player with it).
--
-- Quality: when Python publishes more than one rendition (user-data/oh/qualities, JSON; current one in
-- user-data/oh/quality-current) a quality button appears left of the fullscreen button. It opens a menu (ASS
-- overlay; click / hover, or the "r" key + UP/DOWN/ENTER/ESC). Choosing a rendition freezes playback like a far seek
-- and publishes user-data/oh/quality-request = "<counter>:<format id>" (the counter is shared with seek requests;
-- user-data/oh/quality-pos is the position to resume at). `script-message oh-quality <id|label|height>` does the same.
--
-- Settings (進階設定): gear button left of the quality button, or the "s" key. Two-level menu: Python publishes the list
-- as user-data/oh/settings (JSON [{key,label,hint,kind,value,value_label,options:[{v,label}]}]). Choosing an option
-- of a "restart"/"reload" setting freezes playback like a far seek and publishes user-data/oh/settings-request =
-- "<counter>:<json {key: v} of every setting change not loaded yet>" (+ user-data/oh/settings-pos); "live" settings
-- go to user-data/oh/settings-set = "<seq>:<json>" without a restart. `script-message oh-setting <key> <v>`
-- does the same as the menu.
-- Subtitles: the subtitle icon or "u" opens local track / visibility / scale / delay controls. These use mpv
-- properties directly, without a pipeline restart; track and property observers keep the menu current.
--
-- Info panel (進階資訊): "i" key or the info button. While shown, user-data/oh/info-visible = "1" and Python
-- publishes user-data/oh/info (display-ready JSON: {state, sections:[{title, rows:[{k, v, c}]}]}) about once a
-- second; it is drawn top-left in its own overlay, kept clear of the control bar, the thumbnails and the menus.
-- Rows may carry a trend chart (h = per-second values oldest first, -1 = none; hmax = top of the scale; href = a
-- reference line): drawn right of the value when the whole panel fits at full size with the charts.
--
-- Seekbar: drawn as an ASS overlay (the segment stream has no known length, so the stock OSC is useless).
-- Python provides user-data/oh/duration (total source length) and user-data/oh/start (initial position).
--
-- Opening sources (persistent player): the player cannot open videos by itself (only the main app does, over the
-- single-instance IPC). Dropped files / links are still intercepted (on_load hook, and entries appended to the playlist
-- by --drag-and-drop=append) so mpv never plays a raw, unrestored file: they are discarded with the OSD message
-- "請從主程式開啟影片". Python's own segment loads are recognised by user-data/oh/expect-load (set right
-- before its loadfile / append; old segment entries left in the playlist after a seek tail are removed silently).
-- user-data/oh/internal-urls (JSON URL array) also identifies controller-owned source / feeder URLs so stale playlist
-- entries are discarded silently. Automatic source switches publish user-data/oh/handoff = JSON
-- {token,url,paused,position}; file-loaded checks the actual path and, for playback, waits until time-pos reaches the
-- requested position before publishing handoff-resumed = token. User seek served / resumed keeps its separate meaning.
-- If an intercepted load had already replaced the playing segment, a seek request to the last position brings it
-- back. user-data/oh/active = "0" while no source plays (no seekbar, seeks ignored);
-- user-data/oh/status is a centred status text (idle hint, 解析中…). `script-message oh-reset <tag>` clears all
-- per-source state before Python starts another source and answers with user-data/oh/reset-ack = <tag>.
-- Started with --script-opts=oh-idle=yes, the script begins inactive with the idle hint.
--
-- Casting (投放): cast button left of the info button, or the "c" key. Python publishes the menu as
-- user-data/oh/cast-menu (JSON [{id, label, kind = item|info, hint?, active?}]); opening it publishes
-- user-data/oh/cast-request = "<n>:menu" (Python may rescan), choosing a row "<n>:<id>" (dev:<id>, hls, scan, stop).
-- While user-data/oh/casting = "1" seeks go to Python and the receiver; pause commands also go to the receiver. Quality and
-- settings changes still work: Python answers them with user-data/oh/request-done = <request> instead of a load.
-- `script-message oh-restart <pos> <paused 0|1>` restarts the local segment at pos (always via Python, even inside
-- the cache): used when casting ends.
--
-- Processing queue (背景處理佇列): "a" key or the last row of the settings menu publishes user-data/oh/queue-add =
-- "<n>" (Python sends the source + settings to the queue app and shows 已加入處理佇列 / 處理佇列未開啟).
-- Processed mode: user-data/oh/processed = "1" while Python plays a finished queue output as is (no Lada): seeks are
-- always native mpv seeks (no restart requests; only when a dropped link left mpv empty is a seek request sent, and
-- Python reloads the file there), quality / restart-kind settings only say 已處理的影片 (VSR / smooth stay live),
-- compare works when the file has oh:* video tracks (Python lists them in compare_layers before the load).
--
-- Progressive mode (邊處理邊看): Python plays a queue item while the background worker still processes it, feeding mpv
-- exactly like live mode (same seek-request / served / progress / start / active protocol). user-data/oh/ranges =
-- "" | "a-b,c-d,..." (absolute seconds, the seekbar's timeline) are the parts the queue has already processed: while
-- non-empty they are drawn as a green strip along the bottom of the seekbar, and the hover time of a position outside
-- every range turns amber with 需等待處理 (the seek is a request Python can only serve once the worker got there).
-- Parsed only when the property changes; cleared (with the other per-source state) on oh-reset / active = "0".
--
-- RTX Video Super Resolution: user-data/oh/vsr = "on"|"off" (the "vsr" setting; --script-opts oh-vsr is the
-- startup value). While on and mpv decodes with d3d11va (zero-copy), a labelled filter
-- @oh-vsr:d3d11vpp=scale=<display height / video height, max 4>:scaling-mode=nvidia upscales on the GPU (the NVIDIA
-- driver does the work if RTX Video Enhancement is enabled in the NVIDIA App). display-height comes from mpv, else
-- --script-opts oh-screen-h (Python: primary monitor). user-data/oh/vsr-status is a short status text for the
-- info panel. Smooth playback (video-sync / interpolation) is set by Python directly as mpv properties.

local mp = require 'mp'
local utils = require 'mp.utils'
local msg = require 'mp.msg'

local CACHE_MARGIN = 1.0   -- stay this far from the live edge of a cached range
local SHOW_SECS = 3.0      -- how long the bar stays after a seek / mouse movement
local TICK = 0.12          -- redraw period while the bar is visible (~8 Hz)
local FONT = "Segoe UI"
local FONT_CJK = "Microsoft JhengHei UI"

local counter = 0
local last_request = nil      -- last request string we published
local pending = nil           -- target of a request Python hasn't loaded yet (further seeks are relative to it)
local awaiting = false        -- new segment loaded, waiting for playback to resume
local await_target = nil
local await_served = nil
local initial_loading = true  -- nothing loaded yet (first segment is being produced)
local user_paused = false     -- the user's pause state while a request is in flight
local tail_playing = false    -- user-data/oh/tail: the old timeline keeps playing while the request is prepared
local show_until = 0
-- processed file mode (user-data/oh/processed = "1": Python plays a queue output as is, no Lada) and the "add to
-- queue" request counter; one table = one local (the chunk is near Lua's 200 locals limit)
local proc = { on = false, adds = 0, ranges = {}, ranges_raw = "" }   -- ranges: user-data/oh/ranges parsed [{a, b}]
local mouse = { x = -1, y = -1, hover = false }
local last_mouse_xy = ""
local click_bound = false
local last_draw = 0
local last_data = nil

-- persistent player: source state, opening, status text
local lada_platform = mp.get_opt("oh-platform") or ""
local IDLE_HINT = "請從主程式開啟影片"
local idle_mode = mp.get_opt("oh-idle") == "yes"
local active = not idle_mode      -- a source is loaded / being loaded (user-data/oh/active)
local status_text = nil           -- user-data/oh/status (nil = not published yet)
local internal_active = false     -- one of the controller-owned source / feeder URLs is loaded in mpv
local last_pos = nil              -- last playback position of that controller-owned file
proc.handoff = {
    passthrough = false, pending_ack = nil, last_ack_token = nil,
    compare_passthrough = mp.get_property_native("user-data/oh/passthrough") == "1",
}
local drop_items = {}             -- intercepted loads collected for one open request
local drop_timer = nil

-- quality selection
local qualities = {}              -- [{id=, label=, height=}] high -> low, from user-data/oh/qualities
local quality_current = nil       -- id of the rendition that is playing (or being loaded)
local quality_wanted = nil        -- id requested but not loaded yet
local menu_open = false
local menu_sel = 1                -- highlighted menu row (mouse hover / keys)
local busy_text = "處理中…"       -- what the orange pending text says while a request is in flight
local progress = nil              -- user-data/oh/progress (parsed), see current_progress()

-- advanced settings menu
local settings = {}               -- published by Python (see header)
local settings_wanted = {}        -- key -> option value requested (restart/reload kinds) but not loaded yet
local live_wanted = {}            -- key -> option value sent via settings-set, until Python republishes
local live_seq = 0
local smenu_open = false
local smenu_level = 1             -- 1 = list of settings, 2 = options of settings[smenu_item]
local smenu_item = 1
local smenu_sel = 1               -- highlighted row (index into the rows built by smenu_rows)
local subtitle_menu = false       -- shares settings navigation and rendering, uses mpv properties directly
local subtitle_settings = {}

local function menu_settings()
    return subtitle_menu and subtitle_settings or settings
end

-- casting (投放)
local casting = false             -- user-data/oh/casting == "1"
local cast_rows = {}              -- user-data/oh/cast-menu
local cmenu_open = false
local cmenu_sel = 1
local cast_counter = 0

-- info panel
local info_visible = false
local info_data = nil             -- parsed user-data/oh/info
local info_overlay = mp.create_osd_overlay("ass-events")
info_overlay.z = -1               -- below the control bar / menus overlay
local draw_info                   -- forward declaration

-- volume control (speaker button + slider); volume / mute are persisted by Python (oh_settings.json)
local vol_drag = false            -- left button held on the slider
local vol_osd_text, vol_osd_until = "", 0   -- "音量 NN%" / "靜音" bubble above the speaker button

-- hover preview thumbnails (produced by oh_player.py's ThumbnailWorker, protocol in the header comment of
-- thumbnails below): we publish the hovered time in user-data/oh/thumb-request, Python answers with
-- user-data/oh/thumb-ready = "<requested>|<bgra file>|<w>|<h>|<1 exact / 0 nearest>"
local THUMB_ID = 7            -- overlay-add slot
local THUMB_BUCKET = 2        -- seconds; requests are rounded down to this
local THUMB_FRAC = 0.15       -- displayed thumbnail width as a fraction of the window width
local thumb = nil             -- latest answer from Python { path=, w=, h= }
local thumb_req = ""          -- last published request ("" = nothing hovered)
local thumb_shown = nil       -- key of what overlay-add last drew (nil = overlay removed)
local update_click_binding     -- forward declaration
local set_ranges               -- forward declaration (user-data/oh/ranges)
local current_quality          -- forward declaration
local hide_thumb              -- forward declaration (defined with the other thumbnail helpers)

-- ---------------------------------------------------------------------------------------------------------------
-- helpers
-- ---------------------------------------------------------------------------------------------------------------

local function fmt(t)
    t = math.max(0, math.floor(t or 0))
    local h, m, s = math.floor(t / 3600), math.floor(t % 3600 / 60), t % 60
    if h > 0 then return string.format("%d:%02d:%02d", h, m, s) end
    return string.format("%02d:%02d", m, s)
end

local function fmt_pair(pos, dur)
    -- total length decides the format so both halves line up
    if dur >= 3600 then
        local h, m, s = math.floor(pos / 3600), math.floor(pos % 3600 / 60), math.floor(pos % 60)
        return string.format("%d:%02d:%02d / %s", h, m, s, fmt(dur))
    end
    return fmt(pos) .. " / " .. fmt(dur)
end

local function duration()
    -- the segment streams don't know the source length; Python provides it
    return tonumber((mp.get_property_native("user-data/oh/duration"))) or mp.get_property_number("duration", 0) or 0
end

local function start_pos()
    return tonumber((mp.get_property_native("user-data/oh/start"))) or 0
end

local function seekable_ranges()
    local st = mp.get_property_native("demuxer-cache-state")
    return st and st["seekable-ranges"] or {}
end

local function cached(t)
    for _, r in ipairs(seekable_ranges()) do
        if t >= r["start"] and t <= r["end"] - CACHE_MARGIN then return true end
    end
    return false
end

-- progressive mode: is t inside a part the queue has processed? (true when no ranges are published)
local function processed_at(t)
    local rs = proc.ranges
    if #rs == 0 then return true end
    for i = 1, #rs do
        if t >= rs[i][1] and t <= rs[i][2] then return true end
    end
    return false
end

local function busy()
    return pending ~= nil or ((awaiting or initial_loading) and not casting)
end

local function display_pos()
    if pending then return pending end
    if casting then
        return mp.get_property_number("user-data/oh/cast-position") or start_pos()
    end
    if awaiting and await_target then return await_target end
    if initial_loading then return start_pos() end
    return mp.get_property_number("time-pos") or start_pos()
end

local function display_paused()
    if casting then return mp.get_property_native("user-data/oh/cast-paused") == "1" end
    if pending then return user_paused end
    return mp.get_property_bool("pause", false)
end

-- Python's buffering progress, if it is about the request in flight (a newer request may not be picked up yet)
local function current_progress()
    local p = progress
    if not p or not busy() or casting or type(p.pct) ~= "number" then return nil end
    local req = tostring(p.req or "")
    if pending then
        if req ~= last_request then return nil end
    elseif awaiting then
        if req ~= tostring(await_served or "") then return nil end
    elseif not req:match("^init") then
        return nil
    end
    return p
end

-- the busy text without its ellipsis, for "<text> NN%" (a plain seek / start says 緩衝中)
local function busy_base()
    if busy_text == "處理中…" then return "緩衝中" end
    return (busy_text:gsub("…$", ""))
end

local function busy_label_text()
    local pr = current_progress()
    if pr then return string.format("%s %d%%", busy_base(), pr.pct) end
    return busy_text
end

-- the stage and the remaining time of the request in flight ("修復中", "約 8 秒"; nil when Python has not said)
local function busy_detail()
    local pr = current_progress()
    if not pr then return nil, nil end
    local stage = pr.stage and tostring(pr.stage) or nil
    if stage == "" then stage = nil end
    local eta = type(pr.eta) == "number" and string.format("約 %d 秒", pr.eta) or nil
    return stage, eta
end

-- approximate rendered width of a UTF-8 string at font size `size` (CJK = 1 em, Latin ~0.55 em)
local function text_w(s, size)
    local w, i, n = 0, 1, #s
    while i <= n do
        local c = s:byte(i)
        if c < 0x80 then
            w = w + ((c >= 65 and c <= 90) and 0.64 or c == 32 and 0.3 or 0.55)
            i = i + 1
        elseif c < 0xE0 then
            w = w + 0.6
            i = i + 2
        elseif c < 0xF0 then
            w = w + 1.0
            i = i + 3
        else
            w = w + 1.0
            i = i + 4
        end
    end
    return w * size
end

-- text that may contain ASS control characters (titles, file names)
local function ass_esc(s)
    s = tostring(s or "")
    return (s:gsub("\\", "\\\226\129\160"):gsub("{", "\\{"):gsub("[\r\n]", " "))
end

local function setting_by_key(key)
    for i, it in ipairs(settings) do
        if it.key == key then return it, i end
    end
    return nil
end

-- the option value the UI should show for a setting (a requested one wins over what is in use)
local function setting_shown_value(it)
    return settings_wanted[it.key] or live_wanted[it.key] or it.value
end

local function option_label(it, v)
    if it.kind == "subtitle" and it.value_label and v == it.value then return it.value_label end
    for _, o in ipairs(it.options or {}) do
        if o.v == v then return o.label end
    end
    return tostring(v)
end

-- rows of the settings menu at the current level: {kind = "item"|"action"|"back"|"hint"|"opt", idx =, h = (in units of u)}
local function smenu_rows()
    local rows = {}
    local settings = menu_settings()
    if smenu_level == 1 then
        for i = 1, #settings do rows[#rows + 1] = { kind = "item", idx = i, h = 32 } end
        if not subtitle_menu and active and not proc.on then rows[#rows + 1] = { kind = "action", h = 32 } end   -- add to the queue
    else
        local it = settings[smenu_item]
        if not it then return rows end
        rows[#rows + 1] = { kind = "back", h = 32 }
        if it.hint and it.hint ~= "" then rows[#rows + 1] = { kind = "hint", h = 26 } end
        for i = 1, #(it.options or {}) do rows[#rows + 1] = { kind = "opt", idx = i, h = 30 } end
    end
    return rows
end

-- ---------------------------------------------------------------------------------------------------------------
-- bar geometry / drawing
-- ---------------------------------------------------------------------------------------------------------------

local overlay = mp.create_osd_overlay("ass-events")

current_quality = function()
    for _, q in ipairs(qualities) do
        if q.id == quality_current then return q end
    end
    return qualities[1]
end

-- UI unit: fixed pixel size (only follows the monitor's DPI scaling, not the window size), so buttons and text keep
-- their size when the window is resized; panels that may not fit (info panel) shrink on their own
local function ui_scale()
    local s = mp.get_property_number("display-hidpi-scale", 1) or 1
    return math.max(1, math.min(3, s))
end

local function geom()
    local w, h = mp.get_osd_size()
    if not w or w <= 0 or h <= 0 then return nil end
    local u = ui_scale()
    -- Preserve the normal DPI size until the two button groups would collide.
    local minimum_width = 340
    if #qualities > 1 then
        local cur = current_quality()
        minimum_width = minimum_width + math.max(48, #(cur and cur.label or "") * 0.6 * 15 + 18) + 2
    end
    u = math.min(u, w / minimum_width)
    local bw = 40 * u           -- button hit box width (>= 32 px tall/wide at 720p and scaling with the window)
    local fs_x0 = w - 24 * u + 6 * u - bw
    local g = {
        w = w, h = h, u = u,
        x0 = 24 * u, x1 = w - 24 * u,
        yc = h - 16 * u,        -- bar centre line
        hit_y0 = h - 36 * u,    -- clickable / hoverable bar zone: from here to the window bottom
        -- buttons sit in the label row above the bar: play/pause at the left end, fullscreen at the right end
        by0 = h - 56 * u, by1 = h - 20 * u,
        play_x0 = 24 * u - 6 * u,            play_x1 = 24 * u - 6 * u + bw,
        fs_x1 = w - 24 * u + 6 * u,          fs_x0 = fs_x0,
    }
    -- quality button (pill with the current label) directly left of the fullscreen button; menu above it
    local right = fs_x0
    if #qualities > 1 then
        local cur = current_quality()
        local lw = 0
        for _, q in ipairs(qualities) do lw = math.max(lw, #q.label + (q.mark and 3 or 0)) end   -- (room for the perf dot at the right edge)
        g.q_x1 = fs_x0 - 2 * u
        g.q_x0 = g.q_x1 - math.max(48 * u, (#(cur and cur.label or "") * 0.6 * 15 * u) + 18 * u)
        g.q_cy = (g.by0 + g.by1) / 2
        g.item_h, g.pad = 32 * u, 6 * u
        g.m_w = math.max(96 * u, lw * 0.6 * 16 * u + 52 * u)
        g.m_x1, g.m_x0 = g.q_x1, math.max(4 * u, g.q_x1 - g.m_w)
        g.m_h = #qualities * g.item_h + 2 * g.pad
        g.m_y1 = g.by0 - 2 * u
        g.m_y0 = g.m_y1 - g.m_h
        right = g.q_x0
    end
    -- settings (gear) and info buttons left of that
    g.gear_x1 = right - 2 * u
    g.gear_x0 = g.gear_x1 - bw
    g.sub_x1 = g.gear_x0
    g.sub_x0 = g.sub_x1 - bw
    g.info_x1 = g.sub_x0
    g.info_x0 = g.info_x1 - bw
    g.cast_x1 = g.info_x0
    g.cast_x0 = g.cast_x1 - bw
    g.left_btn_x0 = g.cast_x0
    -- cast menu: right-aligned with the cast button, growing upwards
    if cmenu_open and #cast_rows > 0 then
        local fs, hfs = 15 * u, 12.5 * u
        local mw = 0
        for _, r in ipairs(cast_rows) do
            mw = math.max(mw, text_w(r.label or "", fs) + 48 * u)
            if r.hint and r.hint ~= "" then mw = math.max(mw, text_w(r.hint, hfs) + 40 * u) end
        end
        g.c_fs, g.c_hfs, g.c_pad = fs, hfs, 6 * u
        g.c_w = math.min(w - 8 * u, math.max(200 * u, mw))
        g.c_x0 = math.max(4 * u, g.cast_x1 - g.c_w)
        g.c_x1 = math.min(w - 4 * u, g.c_x0 + g.c_w)
        local total, rows = 0, {}
        for i, r in ipairs(cast_rows) do
            local rh = ((r.hint and r.hint ~= "") and 48 or 32) * u
            rows[i] = { h = rh }
            total = total + rh
        end
        g.c_y1 = g.by0 - 2 * u
        local fit = math.min(1, math.max(0.25, (g.c_y1 - 2 * u - 2 * g.c_pad) / math.max(1, total)))
        g.c_fs, g.c_hfs = fs * fit, hfs * fit
        for _, r in ipairs(rows) do r.h = r.h * fit end
        total = total * fit
        g.c_y0 = math.max(2 * u, g.c_y1 - total - 2 * g.c_pad)
        local y = g.c_y0 + g.c_pad
        for _, r in ipairs(rows) do
            r.y0 = y
            y = y + r.h
            r.y1 = y
        end
        g.c_rows = rows
    end
    -- settings menu: right-aligned with the gear button, growing upwards from the label row
    if smenu_open and #menu_settings() > 0 then
        local settings = menu_settings()
        local fs, hfs = 15 * u, 12.5 * u
        local rows = smenu_rows()
        local mw = 0
        if smenu_level == 1 then
            for _, it in ipairs(settings) do
                mw = math.max(mw, text_w(it.label, fs) + text_w(option_label(it, setting_shown_value(it)) .. " ›", fs) + 64 * u)
            end
            if not subtitle_menu and active and not proc.on then mw = math.max(mw, text_w("加入處理佇列", fs) + text_w("a", fs) + 64 * u) end
        else
            local it = settings[smenu_item] or {}
            mw = text_w("‹ " .. (it.label or ""), fs) + 32 * u
            if it.hint then mw = math.max(mw, text_w(it.hint, hfs) + 28 * u) end
            for _, o in ipairs(it.options or {}) do mw = math.max(mw, text_w(o.label, fs) + 52 * u) end
        end
        g.s_fs, g.s_hfs, g.s_pad = fs, hfs, 6 * u
        g.s_w = math.min(w - 8 * u, math.max(200 * u, mw))
        g.s_x1 = math.min(w - 4 * u, subtitle_menu and g.sub_x1 or g.gear_x1)
        g.s_x0 = math.max(4 * u, g.s_x1 - g.s_w)
        g.s_x1 = g.s_x0 + g.s_w
        local total = 0
        for _, r in ipairs(rows) do total = total + r.h * u end
        g.s_h = total + 2 * g.s_pad
        g.s_y1 = g.by0 - 2 * u
        -- Fit long subtitle track lists into short windows, including keyboard navigation.
        local fit = math.min(1, math.max(0.1, (g.s_y1 - 2 * u - 2 * g.s_pad) / math.max(1, total)))
        g.s_fs, g.s_hfs = fs * fit, hfs * fit
        g.s_h = total * fit + 2 * g.s_pad
        g.s_y0 = math.max(2 * u, g.s_y1 - g.s_h)
        local y = g.s_y0 + g.s_pad
        for _, r in ipairs(rows) do
            r.y0 = y
            y = y + r.h * u * fit
            r.y1 = y
        end
        g.s_rows = rows
    end
    -- left group: play/pause, speaker (mute), volume slider, then the time text. On narrow windows the slider goes
    -- first, then the time text, so nothing ever runs into the right-hand buttons.
    g.spk_x0, g.spk_x1 = g.play_x1, g.play_x1 + bw
    local tfs = 18 * u
    local sample = duration() >= 3600 and "0:00:00 / 0:00:00" or "00:00 / 00:00"
    local time_w = text_w(sample, tfs)
    local limit = g.left_btn_x0 - 8 * u
    g.vol_x0, g.vol_x1 = g.spk_x1 + 4 * u, g.spk_x1 + 4 * u + 64 * u
    g.time_x = g.vol_x1 + 10 * u
    if g.time_x + time_w > limit then
        g.vol_x0, g.vol_x1 = nil, nil
        g.time_x = g.spk_x1 + 2 * u
    end
    g.time_x1 = g.time_x + time_w
    g.show_time = g.time_x1 <= limit
    return g
end

-- row index of the settings menu under the cursor (nil outside the menu; 0 = padding / not selectable)
local function smenu_row_at(g)
    if not smenu_open or not g.s_x0 or mouse.x < 0 or mouse.hover == false then return nil end
    if mouse.x < g.s_x0 or mouse.x > g.s_x1 or mouse.y < g.s_y0 or mouse.y > g.s_y1 then return nil end
    for i, r in ipairs(g.s_rows) do
        if mouse.y >= r.y0 and mouse.y < r.y1 then return r.kind ~= "hint" and i or 0 end
    end
    return 0
end

-- row of the cast menu under the cursor (nil outside; 0 = padding / a row that cannot be chosen)
local function cmenu_row_at(g)
    if not cmenu_open or not g.c_x0 or mouse.x < 0 or mouse.hover == false then return nil end
    if mouse.x < g.c_x0 or mouse.x > g.c_x1 or mouse.y < g.c_y0 or mouse.y > g.c_y1 then return nil end
    for i, r in ipairs(g.c_rows) do
        if mouse.y >= r.y0 and mouse.y < r.y1 then
            return (cast_rows[i] and cast_rows[i].kind == "item") and i or 0
        end
    end
    return 0
end

-- row of the quality menu under the cursor (nil outside; 0 = menu padding)
local function menu_item_at(g)
    if not menu_open or not g.m_x0 or mouse.x < 0 or mouse.hover == false then return nil end
    if mouse.x < g.m_x0 or mouse.x > g.m_x1 or mouse.y < g.m_y0 or mouse.y > g.m_y1 then return nil end
    local i = math.floor((mouse.y - g.m_y0 - g.pad) / g.item_h) + 1
    if i < 1 or i > #qualities then return 0 end
    return i
end

-- what is under the cursor: "play", "fs", "quality", "menu", "bar" or nil (buttons take priority over the bar so a
-- click on them never seeks; while the menu is open the bar is inert)
local function hit(g)
    g = g or geom()
    if not g or mouse.x < 0 or mouse.hover == false then return nil end
    if menu_item_at(g) then return "menu" end
    if smenu_row_at(g) then return "smenu" end
    if cmenu_row_at(g) then return "cmenu" end
    if mouse.y >= g.by0 and mouse.y <= g.by1 then
        if mouse.x >= g.play_x0 and mouse.x <= g.play_x1 then return "play" end
        if mouse.x > g.play_x1 and mouse.x <= g.spk_x1 then return "spk" end
        if g.vol_x0 and mouse.x >= g.vol_x0 - 6 * g.u and mouse.x <= g.vol_x1 + 6 * g.u then return "vol" end
        if mouse.x >= g.fs_x0 and mouse.x <= g.fs_x1 then return "fs" end
        if g.q_x0 and mouse.x >= g.q_x0 and mouse.x <= g.q_x1 then return "quality" end
        if mouse.x >= g.gear_x0 and mouse.x <= g.gear_x1 then return "gear" end
        if mouse.x >= g.sub_x0 and mouse.x <= g.sub_x1 then return "subtitles" end
        if mouse.x >= g.info_x0 and mouse.x <= g.info_x1 then return "info" end
        if mouse.x >= g.cast_x0 and mouse.x <= g.cast_x1 then return "cast" end
    end
    if menu_open or smenu_open or cmenu_open then return nil end
    if duration() <= 0 then return nil end
    if mouse.x >= g.x0 - 12 * g.u and mouse.x <= g.x1 + 12 * g.u and mouse.y >= g.hit_y0 then return "bar" end
    return nil
end

local function in_zone(g)      -- over the seek bar itself
    return hit(g) == "bar"
end

local function x_to_time(g, x)
    local f = (x - g.x0) / (g.x1 - g.x0)
    return math.max(0, math.min(1, f)) * duration()
end

local function visible()
    if not active then return false end
    return busy() or menu_open or smenu_open or cmenu_open or vol_drag or mp.get_property_bool("pause", false)
        or hit() ~= nil
        or mp.get_time() < show_until
end

-- ASS drawing helpers (coordinates are window pixels because res_x/res_y = osd size)
local function rect(x0, y0, x1, y1, bgr, alpha)
    return string.format("{\\an7\\pos(0,0)\\bord0\\shad0\\blur0\\1c&H%s&\\1a&H%s&\\p1}m %.1f %.1f l %.1f %.1f %.1f %.1f %.1f %.1f{\\p0}",
        bgr, alpha, x0, y0, x1, y0, x1, y1, x0, y1)
end

local function poly(pts, bgr, alpha)
    local s = {}
    for i, p in ipairs(pts) do s[#s + 1] = string.format("%s %.1f %.1f", i == 1 and "m" or "l", p[1], p[2]) end
    return string.format("{\\an7\\pos(0,0)\\bord0\\shad0\\blur0\\1c&H%s&\\1a&H%s&\\p1}%s{\\p0}",
        bgr, alpha, table.concat(s, " "))
end

local function circle(cx, cy, r, bgr, alpha)
    local k = 0.5523 * r
    return string.format("{\\an7\\pos(0,0)\\bord0\\shad0\\blur0\\1c&H%s&\\1a&H%s&\\p1}" ..
        "m %.1f %.1f b %.1f %.1f %.1f %.1f %.1f %.1f b %.1f %.1f %.1f %.1f %.1f %.1f " ..
        "b %.1f %.1f %.1f %.1f %.1f %.1f b %.1f %.1f %.1f %.1f %.1f %.1f{\\p0}",
        bgr, alpha,
        cx - r, cy, cx - r, cy - k, cx - k, cy - r, cx, cy - r,
        cx + k, cy - r, cx + r, cy - k, cx + r, cy,
        cx + r, cy + k, cx + k, cy + r, cx, cy + r,
        cx - k, cy + r, cx - r, cy + k, cx - r, cy)
end

-- quality-menu perf verdict dots (BGR): live green, prebuffer amber, queue red
local PERF_MARK_BGR = {live = "50C850", prebuffer = "30B4F0", queue = "5050E6"}

local function text(x, y, an, size, str, bgr, font)
    return string.format("{\\an%d\\pos(%.1f,%.1f)\\fn%s\\fs%.1f\\b0\\bord1.6\\shad0\\blur0\\1c&H%s&\\3c&H000000&\\1a&H00&\\q2}%s",
        an, x, y, font or FONT, size, bgr or "FFFFFF", str)
end

-- hollow rectangle of line width t
local function frame(x0, y0, x1, y1, t, bgr, alpha)
    return table.concat({
        rect(x0, y0, x1, y0 + t, bgr, alpha), rect(x0, y1 - t, x1, y1, bgr, alpha),
        rect(x0, y0 + t, x0 + t, y1 - t, bgr, alpha), rect(x1 - t, y0 + t, x1, y1 - t, bgr, alpha),
    }, "\n")
end

-- filled rectangle with rounded corners (radius r)
local function rrect(x0, y0, x1, y1, r, bgr, alpha)
    r = math.max(0, math.min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    local k = r * (1 - 0.5523)
    return string.format("{\\an7\\pos(0,0)\\bord0\\shad0\\blur0\\1c&H%s&\\1a&H%s&\\p1}" ..
        "m %.1f %.1f l %.1f %.1f b %.1f %.1f %.1f %.1f %.1f %.1f l %.1f %.1f b %.1f %.1f %.1f %.1f %.1f %.1f " ..
        "l %.1f %.1f b %.1f %.1f %.1f %.1f %.1f %.1f l %.1f %.1f b %.1f %.1f %.1f %.1f %.1f %.1f{\\p0}",
        bgr, alpha,
        x0 + r, y0, x1 - r, y0, x1 - k, y0, x1, y0 + k, x1, y0 + r,
        x1, y1 - r, x1, y1 - k, x1 - k, y1, x1 - r, y1,
        x0 + r, y1, x0 + k, y1, x0, y1 - k, x0, y1 - r,
        x0, y0 + r, x0, y0 + k, x0 + k, y0, x0 + r, y0)
end

-- plain text without outline (for text on the dark panels); `bold` optional
local function ptext(x, y, an, size, str, bgr, bold)
    return string.format("{\\an%d\\pos(%.1f,%.1f)\\fn%s\\fs%.1f\\b%d\\bord0\\shad0\\blur0\\1c&H%s&\\1a&H00&\\q2}%s",
        an, x, y, FONT_CJK, size, bold and 1 or 0, bgr or "FFFFFF", str)
end

-- gear icon: 8 teeth around a hub, hole drawn in dark grey
local function gear_icon(cx, cy, u)
    local pts = {}
    local ro, ri = 8.8 * u, 6.4 * u
    for k = 0, 7 do
        local a = k * math.pi / 4
        for _, p in ipairs({ { ri, a - 0.30 }, { ro, a - 0.17 }, { ro, a + 0.17 }, { ri, a + 0.30 } }) do
            pts[#pts + 1] = { cx + p[1] * math.cos(p[2]), cy + p[1] * math.sin(p[2]) }
        end
    end
    return poly(pts, "FFFFFF", "00") .. "\n" .. circle(cx, cy, 2.9 * u, "303030", "00")
end

-- thick line segment from (x0,y0) to (x1,y1)
local function seg(x0, y0, x1, y1, t, bgr, alpha)
    local dx, dy = x1 - x0, y1 - y0
    local l = math.sqrt(dx * dx + dy * dy)
    if l <= 0 then return "" end
    local nx, ny = -dy / l * t / 2, dx / l * t / 2
    return poly({ { x0 + nx, y0 + ny }, { x1 + nx, y1 + ny }, { x1 - nx, y1 - ny }, { x0 - nx, y0 - ny } }, bgr, alpha)
end

-- arc of radius r and thickness t between angles a0..a1 (radians, 0 = right, clockwise on screen)
local function arc(cx, cy, r, t, a0, a1, bgr, alpha)
    local pts, n = {}, 10
    for i = 0, n do
        local a = a0 + (a1 - a0) * i / n
        pts[#pts + 1] = { cx + (r + t / 2) * math.cos(a), cy + (r + t / 2) * math.sin(a) }
    end
    for i = n, 0, -1 do
        local a = a0 + (a1 - a0) * i / n
        pts[#pts + 1] = { cx + (r - t / 2) * math.cos(a), cy + (r - t / 2) * math.sin(a) }
    end
    return poly(pts, bgr, alpha)
end

-- speaker: muted = cross, < 50 % one wave, otherwise two waves
local function speaker_icon(cx, cy, u, muted, vol)
    local x = cx - 4 * u
    local p = { poly({ { x - 7 * u, cy - 3.5 * u }, { x - 3 * u, cy - 3.5 * u }, { x + 2 * u, cy - 8 * u },
        { x + 2 * u, cy + 8 * u }, { x - 3 * u, cy + 3.5 * u }, { x - 7 * u, cy + 3.5 * u } }, "FFFFFF", "00") }
    if muted or vol <= 0 then
        p[#p + 1] = seg(x + 5.5 * u, cy - 4 * u, x + 12.5 * u, cy + 4 * u, 2 * u, "FFFFFF", "00")
        p[#p + 1] = seg(x + 5.5 * u, cy + 4 * u, x + 12.5 * u, cy - 4 * u, 2 * u, "FFFFFF", "00")
    else
        p[#p + 1] = arc(x + 2 * u, cy, 4.8 * u, 1.9 * u, -0.85, 0.85, "FFFFFF", "00")
        if vol >= 50 then p[#p + 1] = arc(x + 2 * u, cy, 9 * u, 1.9 * u, -0.9, 0.9, "FFFFFF", "00") end
    end
    return table.concat(p, "\n")
end

-- info icon: white disc with a dark "i"
local function info_icon(cx, cy, u)
    return table.concat({
        circle(cx, cy, 9 * u, "FFFFFF", "00"),
        rect(cx - 1.3 * u, cy - 1.8 * u, cx + 1.3 * u, cy + 5.6 * u, "303030", "00"),
        circle(cx, cy - 4.6 * u, 1.5 * u, "303030", "00"),
    }, "\n")
end

-- cast icon: a screen with a triangle under it (orange while casting)
local function cast_icon(cx, cy, u, on)
    local col = on and "00B4FF" or "FFFFFF"
    return table.concat({
        rect(cx - 9.5 * u, cy - 8 * u, cx + 9.5 * u, cy - 6.2 * u, col, "00"),
        rect(cx - 9.5 * u, cy - 6.2 * u, cx - 7.7 * u, cy + 4.5 * u, col, "00"),
        rect(cx + 7.7 * u, cy - 6.2 * u, cx + 9.5 * u, cy + 4.5 * u, col, "00"),
        rect(cx - 9.5 * u, cy + 2.7 * u, cx - 4.5 * u, cy + 4.5 * u, col, "00"),
        rect(cx + 4.5 * u, cy + 2.7 * u, cx + 9.5 * u, cy + 4.5 * u, col, "00"),
        poly({ { cx - 5.5 * u, cy + 9 * u }, { cx + 5.5 * u, cy + 9 * u }, { cx, cy + 2.2 * u } }, col, "00"),
    }, "\n")
end

-- Where the thumbnail goes: centred on the cursor, clamped to the window, bottom edge at `bottom`.
local function thumb_geometry(g, mx, bottom)
    if not thumb then return nil end
    local dw = math.floor(math.max(120, math.min(thumb.w * 1.75, g.w * THUMB_FRAC)) / 2 + 0.5) * 2
    local dh = math.floor(dw * thumb.h / thumb.w / 2 + 0.5) * 2
    local margin = 8 * g.u
    local x = math.floor(math.max(margin, math.min(g.w - dw - margin, mx - dw / 2)) + 0.5)
    local y = math.floor(bottom - dh + 0.5)
    if y < 0 then return nil end
    return { x = x, y = y, dw = dw, dh = dh, path = thumb.path, w = thumb.w, h = thumb.h }
end

local function show_thumb(th)
    local key = string.format("%s|%d|%d|%d|%d", th.path, th.x, th.y, th.dw, th.dh)
    if key == thumb_shown then return end
    -- args: id x y file offset fmt w h stride dw dh (dw/dh scale the bitmap to the display size)
    local ok = pcall(mp.commandv, "overlay-add", tostring(THUMB_ID), tostring(th.x), tostring(th.y), th.path, "0", "bgra",
        tostring(th.w), tostring(th.h), tostring(th.w * 4), tostring(th.dw), tostring(th.dh))
    thumb_shown = ok and key or nil
end

local function publish_thumb_request(secs)
    local v = secs and string.format("%d", secs) or ""
    if v ~= thumb_req then
        thumb_req = v
        mp.set_property("user-data/oh/thumb-request", v)
    end
end

-- hide_thumb(true) also tells Python nothing is hovered any more
hide_thumb = function(clear_request)
    if thumb_shown then
        mp.commandv("overlay-remove", tostring(THUMB_ID))
        thumb_shown = nil
    end
    if clear_request then
        publish_thumb_request(nil)
        thumb = nil       -- Python re-answers (with a new sequence number) when the next hover starts
    end
end

local function draw()
    local function fit_menu_text(s, size, width)
        if text_w(s, size) <= width then return s end
        while s ~= "" and text_w(s .. "…", size) > width do
            s = s:gsub("[^\128-\191][\128-\191]*$", "")
        end
        return s .. "…"
    end
    local g = geom()
    if not g then return end
    local u, w, h = g.u, g.w, g.h
    local dur = duration()
    local pos = display_pos()
    if dur > 0 then pos = math.max(0, math.min(dur, pos)) end
    local hot = in_zone(g)
    local busy_now = busy()
    local p = {}
    local function add(s) p[#p + 1] = s end

    -- background: real gradient so text stays readable on bright video. One rectangle with a large gaussian blur:
    -- its top edge fades smoothly from transparent (h - 62u) to full (h - 26u); it overhangs the window on the
    -- other three sides so only the top edge shows the blur.
    do
        local edge, blur = h - 44 * u, 18 * u
        local ox = 3 * blur
        add(string.format("{\\an7\\pos(0,0)\\bord0\\shad0\\blur%.1f\\1c&H000000&\\1a&H50&\\p1}" ..
            "m %.1f %.1f l %.1f %.1f %.1f %.1f %.1f %.1f{\\p0}",
            blur, -ox, edge, w + ox, edge, w + ox, h + ox, -ox, h + ox))
    end

    local thick = (hot and 9 or 5) * u
    local top, bot = g.yc - thick / 2, g.yc + thick / 2
    add(rect(g.x0, top, g.x1, bot, "FFFFFF", "C8"))  -- full-length track

    local function xt(t) return g.x0 + (g.x1 - g.x0) * math.max(0, math.min(1, t / dur)) end

    if dur > 0 then
        for _, r in ipairs(seekable_ranges()) do
            local xs, xe = xt(r["start"]), xt(r["end"])
            if xe - xs < 2 then xe = xs + 2 end
            add(rect(xs, top, xe, bot, "FFA050", "20"))  -- processed / cached (RGB 50A0FF)
        end
        -- progressive mode: parts the queue has processed (green strip along the bottom of the bar, RGB 5CD68A)
        local y0 = bot - (bot - top) * 0.45
        for _, r in ipairs(proc.ranges) do
            local xs, xe = xt(r[1]), xt(r[2])
            if xe - xs < 2 then xe = xs + 2 end
            add(rect(xs, y0, xe, bot, "8AD65C", "10"))
        end
    end

    -- left: play/pause button, then position / total
    local fs = 18 * u
    local ty = h - 27 * u
    local cy_btn = (g.by0 + g.by1) / 2
    local over = hit(g)
    local paused = display_paused()
    do
        local cx = (g.play_x0 + g.play_x1) / 2
        if over == "play" then add(circle(cx, cy_btn, 17 * u, "FFFFFF", "A0")) end
        if paused then      -- shows the action: play triangle while paused
            add(poly({ { cx - 5 * u, cy_btn - 8 * u }, { cx - 5 * u, cy_btn + 8 * u }, { cx + 8 * u, cy_btn } }, "FFFFFF", "00"))
        else                -- pause bars while playing
            add(rect(cx - 6 * u, cy_btn - 7.5 * u, cx - 2 * u, cy_btn + 7.5 * u, "FFFFFF", "00"))
            add(rect(cx + 2 * u, cy_btn - 7.5 * u, cx + 6 * u, cy_btn + 7.5 * u, "FFFFFF", "00"))
        end
    end
    -- speaker (mute toggle) + volume slider
    do
        local cx = (g.spk_x0 + g.spk_x1) / 2
        local muted = mp.get_property_bool("mute", false)
        local vol = mp.get_property_number("volume", 100) or 100
        if over == "spk" then add(circle(cx, cy_btn, 17 * u, "FFFFFF", "A0")) end
        add(speaker_icon(cx, cy_btn, u, muted, vol))
        if g.vol_x0 then
            local x0, x1 = g.vol_x0, g.vol_x1
            local f = math.max(0, math.min(1, vol / 100))
            local th = (over == "vol" or vol_drag) and 5 * u or 4 * u
            add(rect(x0, cy_btn - th / 2, x1, cy_btn + th / 2, "FFFFFF", "B0"))
            add(rect(x0, cy_btn - th / 2, x0 + (x1 - x0) * f, cy_btn + th / 2, muted and "909090" or "FFFFFF", "00"))
            add(circle(x0 + (x1 - x0) * f, cy_btn, ((over == "vol" or vol_drag) and 6.5 or 5) * u,
                muted and "909090" or "FFFFFF", "00"))
        end
    end
    if mp.get_time() < vol_osd_until and vol_osd_text ~= "" then
        local bfs = 15 * u
        local bw2 = text_w(vol_osd_text, bfs) + 18 * u
        local bx = math.max(4 * u, (g.spk_x0 + g.spk_x1) / 2 - bw2 / 2)
        local by = g.by0 - 8 * u
        add(rrect(bx, by - 26 * u, bx + bw2, by, 6 * u, "000000", "40"))
        add(ptext(bx + bw2 / 2, by - 13 * u, 5, bfs, vol_osd_text, "FFFFFF"))
    end
    -- position / total (dropped while the orange busy text needs the room, or when the window is too narrow)
    -- the busy label carries "<text> NN% · <stage> · 約 N 秒  <pos>"; on a narrow window the stage goes first, then the ETA
    local busy_label, busy_x0 = nil, math.huge
    if busy_now then
        local base, stage, eta = busy_label_text(), busy_detail()
        local variants = {}
        if stage and eta then variants[#variants + 1] = base .. " · " .. stage .. " · " .. eta end
        if eta then variants[#variants + 1] = base .. " · " .. eta end
        variants[#variants + 1] = base
        for _, v in ipairs(variants) do
            busy_label = v .. "  " .. fmt(pos)
            busy_x0 = g.left_btn_x0 - 6 * u - text_w(busy_label, fs)
            if busy_x0 > (g.show_time and g.time_x1 or (g.vol_x1 or g.spk_x1)) + 6 * u then break end
        end
    end
    if g.show_time and busy_x0 > g.time_x1 + 6 * u then
        add(text(g.time_x, ty, 1, fs, fmt_pair(pos, dur)))
    end

    -- right: fullscreen button (corner brackets; inward-pointing while fullscreen)
    do
        local cx = (g.fs_x0 + g.fs_x1) / 2
        if over == "fs" then add(circle(cx, cy_btn, 17 * u, "FFFFFF", "A0")) end
        local is_fs = mp.get_property_bool("fullscreen", false)
        local arm, th = 5.5 * u, 2 * u
        for _, sx in ipairs({ -1, 1 }) do
            for _, sy in ipairs({ -1, 1 }) do
                local off = (is_fs and 3 or 8) * u
                local px, py = cx + sx * off, cy_btn + sy * off
                local dir = is_fs and 1 or -1       -- windowed: arms point inwards from far corners; fullscreen: outwards
                local ax = px + sx * dir * arm
                local ay = py + sy * dir * arm
                add(rect(math.min(px, ax), py - th / 2, math.max(px, ax), py + th / 2, "FFFFFF", "00"))
                add(rect(px - th / 2, math.min(py, ay), px + th / 2, math.max(py, ay), "FFFFFF", "00"))
            end
        end
    end

    -- quality button: pill with the current label (only when there is something to choose from)
    if g.q_x0 then
        local cur = current_quality()
        local lit = over == "quality" or menu_open
        local y0, y1 = g.q_cy - 12 * u, g.q_cy + 12 * u
        if lit then add(rect(g.q_x0, y0, g.q_x1, y1, "FFFFFF", "A0")) end
        add(frame(g.q_x0, y0, g.q_x1, y1, 1.6 * u, "FFFFFF", lit and "20" or "60"))
        add(text((g.q_x0 + g.q_x1) / 2, g.q_cy, 5, 15 * u, cur and cur.label or "", "FFFFFF"))
    end

    -- settings (gear) and info buttons
    do
        local cx = (g.gear_x0 + g.gear_x1) / 2
        if over == "gear" or (smenu_open and not subtitle_menu) then add(circle(cx, cy_btn, 17 * u, "FFFFFF", over == "gear" and "A0" or "C0")) end
        add(gear_icon(cx, cy_btn, u))
        cx = (g.sub_x0 + g.sub_x1) / 2
        if over == "subtitles" or (smenu_open and subtitle_menu) then
            add(circle(cx, cy_btn, 17 * u, "FFFFFF", "A0"))
        end
        -- Language-independent subtitle icon: a screen with two segmented caption lines.
        add(frame(cx - 10 * u, cy_btn - 8 * u, cx + 10 * u, cy_btn + 8 * u, 1.7 * u, "FFFFFF", "00"))
        add(rect(cx - 6 * u, cy_btn - 2 * u, cx - 3 * u, cy_btn, "FFFFFF", "00"))
        add(rect(cx - 1 * u, cy_btn - 2 * u, cx + 6 * u, cy_btn, "FFFFFF", "00"))
        add(rect(cx - 6 * u, cy_btn + 2 * u, cx + 1 * u, cy_btn + 4 * u, "FFFFFF", "00"))
        add(rect(cx + 3 * u, cy_btn + 2 * u, cx + 6 * u, cy_btn + 4 * u, "FFFFFF", "00"))
        cx = (g.info_x0 + g.info_x1) / 2
        if over == "info" or info_visible then add(circle(cx, cy_btn, 17 * u, "FFFFFF", over == "info" and "A0" or "C0")) end
        add(info_icon(cx, cy_btn, u))
        cx = (g.cast_x0 + g.cast_x1) / 2
        if over == "cast" or cmenu_open then add(circle(cx, cy_btn, 17 * u, "FFFFFF", over == "cast" and "A0" or "C0")) end
        add(cast_icon(cx, cy_btn, u, casting))
    end

    -- right text while a restart is in flight
    if busy_now then
        local left = g.vol_x1 or g.spk_x1
        local bfs = fs
        if busy_x0 < left + 6 * u then     -- very narrow window: shrink rather than run into the volume slider
            bfs = fs * math.max(0.5, (g.left_btn_x0 - 12 * u - left) / text_w(busy_label, fs))
        end
        add(string.format("{\\an3\\pos(%.1f,%.1f)\\fn%s\\fs%.1f\\b0\\bord1.6\\shad0\\1c&H00B4FF&\\3c&H000000&\\q2}%s",
            g.left_btn_x0 - 6 * u, ty, FONT_CJK, bfs, busy_label))
    end

    -- playhead (orange while the target is not playing yet)
    if dur > 0 then
        local hx = xt(pos)
        add(circle(hx, g.yc, (hot and 8.5 or 7) * u, busy_now and "00A0FF" or "FFFFFF", "00"))
    end

    -- hover marker + time tooltip
    if hot and dur > 0 then
        local mx = math.max(g.x0, math.min(g.x1, mouse.x))
        add(rect(mx - 1 * u, g.yc - 8 * u, mx + 1 * u, g.yc + 8 * u, "FFFFFF", "60"))
        local t = x_to_time(g, mx)
        publish_thumb_request(math.floor(t / THUMB_BUCKET) * THUMB_BUCKET)
        local label = fmt_pair(t, 0):gsub(" / .*", "")
        if dur >= 3600 then
            local hh, mm, ss = math.floor(t / 3600), math.floor(t % 3600 / 60), math.floor(t % 60)
            label = string.format("%d:%02d:%02d", hh, mm, ss)
        end
        local tfs = 16 * u
        local wait = #proc.ranges > 0 and not processed_at(t)   -- progressive mode: not processed yet
        if wait then label = label .. " 需等待處理" end
        local tw = (wait and text_w(label, tfs) or (#label * 0.58 * tfs)) + 16 * u
        local cx = math.max(tw / 2 + 4 * u, math.min(w - tw / 2 - 4 * u, mx))
        local ty2 = h - 66 * u
        add(rect(cx - tw / 2, ty2 - 14 * u, cx + tw / 2, ty2 + 14 * u, "000000", "40"))
        -- amber = outside the processed range (needs a restart), white = instant
        add(text(cx, ty2, 5, tfs, label, (cached(t) and not wait) and "FFFFFF" or "40C8FF"))

        -- preview thumbnail above the tooltip (bitmap overlay), framed by four thin ASS rectangles
        local th = thumb_geometry(g, mx, ty2 - 14 * u - 6 * u)
        if th then
            local b = 2 * u
            add(rect(th.x - b, th.y - b, th.x + th.dw + b, th.y, "FFFFFF", "30"))
            add(rect(th.x - b, th.y + th.dh, th.x + th.dw + b, th.y + th.dh + b, "FFFFFF", "30"))
            add(rect(th.x - b, th.y, th.x, th.y + th.dh, "FFFFFF", "30"))
            add(rect(th.x + th.dw, th.y, th.x + th.dw + b, th.y + th.dh, "FFFFFF", "30"))
            show_thumb(th)
        else
            hide_thumb(false)
        end
    else
        hide_thumb(true)
    end

    -- quality menu (drawn last, on top): dark panel, hover row highlighted, playing one marked with an orange dot
    if menu_open and g.m_x0 then
        add(rect(g.m_x0 - 2 * u, g.m_y0 - 2 * u, g.m_x1 + 2 * u, g.m_y1 + 2 * u, "000000", "50"))   -- soft edge
        add(rect(g.m_x0, g.m_y0, g.m_x1, g.m_y1, "262626", "10"))
        add(frame(g.m_x0, g.m_y0, g.m_x1, g.m_y1, 1.2 * u, "FFFFFF", "70"))
        for i, q in ipairs(qualities) do
            local y0 = g.m_y0 + g.pad + (i - 1) * g.item_h
            local cy = y0 + g.item_h / 2
            local playing = q.id == quality_current
            if i == menu_sel then add(rect(g.m_x0 + 3 * u, y0, g.m_x1 - 3 * u, y0 + g.item_h, "FFFFFF", "B0")) end
            if playing then add(circle(g.m_x0 + 18 * u, cy, 4.5 * u, "00A0FF", "00")) end
            add(text(g.m_x0 + 32 * u, cy, 4, 16 * u, q.label, playing and "00B4FF" or "FFFFFF"))
            local mc = PERF_MARK_BGR[q.mark or ""]   -- q.mark: perf verdict level (perf_hint.py), a dot at the right edge
            if mc then add(circle(g.m_x1 - 16 * u, cy, 4 * u, mc, "00")) end
        end
    end

    -- settings menu: same look as the quality menu
    if smenu_open and g.s_x0 then
        local settings = menu_settings()
        add(rect(g.s_x0 - 2 * u, g.s_y0 - 2 * u, g.s_x1 + 2 * u, g.s_y1 + 2 * u, "000000", "50"))
        add(rect(g.s_x0, g.s_y0, g.s_x1, g.s_y1, "262626", "10"))
        add(frame(g.s_x0, g.s_y0, g.s_x1, g.s_y1, 1.2 * u, "FFFFFF", "70"))
        local it2 = settings[smenu_item]
        for i, r in ipairs(g.s_rows) do
            local cy = (r.y0 + r.y1) / 2
            if i == smenu_sel and r.kind ~= "hint" then
                add(rect(g.s_x0 + 3 * u, r.y0, g.s_x1 - 3 * u, r.y1, "FFFFFF", "B0"))
            end
            if r.kind == "item" then
                local it = settings[r.idx]
                local v = setting_shown_value(it)
                local pend = v ~= it.value
                add(text(g.s_x0 + 14 * u, cy, 4, g.s_fs, ass_esc(it.label), "FFFFFF", FONT_CJK))
                local label = option_label(it, v)
                if subtitle_menu then label = fit_menu_text(label, g.s_fs, g.s_w - text_w(it.label, g.s_fs) - 64 * u) end
                add(text(g.s_x1 - 12 * u, cy, 6, g.s_fs, ass_esc(label) .. (pend and "…" or "") .. "  ›",
                    pend and "00B4FF" or "D0D0D0", FONT_CJK))
            elseif r.kind == "action" then
                add(rect(g.s_x0 + 8 * u, r.y0, g.s_x1 - 8 * u, r.y0 + 1 * u, "FFFFFF", "A0"))
                add(text(g.s_x0 + 14 * u, cy, 4, g.s_fs, "加入處理佇列", "00B4FF", FONT_CJK))
                add(text(g.s_x1 - 12 * u, cy, 6, g.s_fs, "a", "A8A8A8", FONT_CJK))
            elseif r.kind == "back" and it2 then
                add(text(g.s_x0 + 12 * u, cy, 4, g.s_fs, "‹  " .. ass_esc(it2.label), "FFFFFF", FONT_CJK))
                add(rect(g.s_x0 + 8 * u, r.y1 - 1 * u, g.s_x1 - 8 * u, r.y1, "FFFFFF", "A0"))
            elseif r.kind == "hint" and it2 then
                add(text(g.s_x0 + 14 * u, cy, 4, g.s_hfs, ass_esc(fit_menu_text(it2.hint, g.s_hfs, g.s_w - 28 * u)), "A8A8A8", FONT_CJK))
            elseif r.kind == "opt" and it2 then
                local o = it2.options[r.idx]
                local cur = o.v == setting_shown_value(it2)
                if cur then add(circle(g.s_x0 + 18 * u, cy, 4.5 * u, "00A0FF", "00")) end
                add(text(g.s_x0 + 32 * u, cy, 4, g.s_fs, ass_esc(fit_menu_text(o.label, g.s_fs, g.s_w - 46 * u)), cur and "00B4FF" or "FFFFFF", FONT_CJK))
            end
        end
    end

    -- cast menu: same look; rows that cannot be chosen are grey, the current cast state orange
    if cmenu_open and g.c_x0 then
        add(rect(g.c_x0 - 2 * u, g.c_y0 - 2 * u, g.c_x1 + 2 * u, g.c_y1 + 2 * u, "000000", "50"))
        add(rect(g.c_x0, g.c_y0, g.c_x1, g.c_y1, "262626", "10"))
        add(frame(g.c_x0, g.c_y0, g.c_x1, g.c_y1, 1.2 * u, "FFFFFF", "70"))
        for i, r in ipairs(g.c_rows) do
            local row = cast_rows[i]
            local item = row.kind == "item"
            local has_hint = row.hint and row.hint ~= ""
            if item and i == cmenu_sel then add(rect(g.c_x0 + 3 * u, r.y0, g.c_x1 - 3 * u, r.y1, "FFFFFF", "B0")) end
            local cy = has_hint and (r.y0 + 16 * u) or (r.y0 + r.y1) / 2
            local col = row.active and "00B4FF" or (item and "FFFFFF" or "A8A8A8")
            if row.active then add(circle(g.c_x0 + 16 * u, cy, 4.5 * u, "00A0FF", "00")) end
            add(text(g.c_x0 + (row.active and 28 or 14) * u, cy, 4, g.c_fs, ass_esc(row.label), col, FONT_CJK))
            if has_hint then
                add(text(g.c_x0 + 14 * u, r.y0 + 35 * u, 4, g.c_hfs, ass_esc(row.hint), "8C8C8C", FONT_CJK))
            end
        end
    end

    -- centre: buffering progress, only for the initial load while nothing is on screen yet (seeks / switches report in
    -- the bar above and leave the picture free); a centred status text (解析中…) has priority. Without a percentage
    -- yet the bar slides.
    if busy_now and initial_loading and not casting and mp.get_property_native("video-params") == nil
        and (status_text == nil or status_text == "") then
        do
            local now = mp.get_time()
            local pr = current_progress()
            local title = busy_label_text()
            local stage, eta = busy_detail()
            local sub = stage
            if eta then sub = (sub and (sub .. " · ") or "") .. eta end
            local fs1, fs2 = 21 * u, 14 * u
            local bw = math.max(250 * u, text_w(title, fs1) + 48 * u, sub and (text_w(sub, fs2) + 48 * u) or 0)
            bw = math.min(bw, w - 16 * u)
            local bh = (sub and 90 or 70) * u
            local x0, y0 = (w - bw) / 2, (h - bh) / 2
            add(rrect(x0, y0, x0 + bw, y0 + bh, 10 * u, "141414", "60"))
            add(ptext(w / 2, y0 + 25 * u, 5, fs1, ass_esc(title), "FFFFFF", true))
            local bx0, bx1, cy, th = x0 + 24 * u, x0 + bw - 24 * u, y0 + 49 * u, 5 * u
            add(rrect(bx0, cy - th / 2, bx1, cy + th / 2, th / 2, "FFFFFF", "C0"))
            local f0, f1
            if pr then
                f0, f1 = 0, math.max(0, math.min(1, pr.pct / 100))
            else
                local f = (now * 0.7) % 1.3
                f0, f1 = math.max(0, f - 0.3), math.min(1, f)
            end
            if f1 - f0 > 0.005 then
                add(rrect(bx0 + (bx1 - bx0) * f0, cy - th / 2, bx0 + (bx1 - bx0) * f1, cy + th / 2, th / 2,
                    "00B4FF", "00"))
            end
            if sub then add(ptext(w / 2, y0 + 71 * u, 5, fs2, ass_esc(sub), "B4B4B4")) end
        end
    end

    local data = table.concat(p, "\n")
    overlay.res_x, overlay.res_y = w, h
    if data ~= last_data or overlay.res_x ~= w then
        overlay.data = data
        overlay:update()
        last_data = data
    end
    last_draw = mp.get_time()
end

-- ---------------------------------------------------------------------------------------------------------------
-- redraw scheduling: a timer runs only while the bar is visible
-- ---------------------------------------------------------------------------------------------------------------

local timer
local timer_running = false

local function finish_await()
    if not awaiting then return end
    awaiting = false
    mp.set_property("user-data/oh/resumed", await_served or "")
    show_until = math.max(show_until, mp.get_time() + SHOW_SECS)
end

local function tick()
    if awaiting and mp.get_property_native("core-idle") == false
        and not mp.get_property_bool("seeking", false) and mp.get_property_number("time-pos") then
        finish_await()
    end
    if visible() then
        draw()
    else
        overlay:remove()
        hide_thumb(true)
        update_click_binding()
        last_data = nil
        timer:stop()
        timer_running = false
    end
end

timer = mp.add_periodic_timer(TICK, tick)
timer:stop()

local function request_draw(throttle)
    if not timer_running then
        timer:resume()
        timer_running = true
    end
    if not throttle or mp.get_time() - last_draw > 0.04 then
        if visible() then draw() end
    end
end

local function show(secs)
    show_until = math.max(show_until, mp.get_time() + (secs or SHOW_SECS))
    request_draw()
end

-- ---------------------------------------------------------------------------------------------------------------
-- seeking
-- ---------------------------------------------------------------------------------------------------------------

-- first request of a chain: remember the user's own pause state (keep-open also pauses at the end of a segment; that
-- is not a user pause). freeze: pause the current playback right away (quality / settings changes, restarts); plain
-- seeks leave it to Python, which keeps the old timeline playing when it can (see header)
local function begin_request(freeze)
    proc.handoff.cancel()
    if not pending then
        user_paused = mp.get_property_bool("pause", false) and not mp.get_property_bool("eof-reached", false)
    end
    if freeze then mp.set_property_bool("pause", true) end
end

local function seek_to(t)
    if not active then return end
    proc.handoff.cancel()
    local dur = duration()
    if dur > 0 then t = math.min(t, dur - 1) end
    t = math.max(0, t)
    if casting then
        counter = counter + 1
        mp.set_property("user-data/oh/seek-request", string.format("%d:%.3f", counter, t))
        show(SHOW_SECS)
        return
    end
    -- a processed file is plain mpv playback: seeks are always native (unless mpv has nothing loaded: a dropped link
    -- replaced the file, Python then reloads it at t)
    local native = proc.on and not mp.get_property_bool("idle-active", false)
    if not pending and (native or cached(t)) then
        mp.commandv("seek", string.format("%.3f", t), "absolute+exact")
    else
        begin_request(false)
        counter = counter + 1
        pending = t
        awaiting = false
        busy_text = "處理中…"
        last_request = string.format("%d:%.3f", counter, t)
        mp.set_property("user-data/oh/seek-request", last_request)
    end
    show(SHOW_SECS)
end

local function seek_by(delta)
    seek_to(display_pos() + delta)
end

-- add the playing source to the background processing queue: Python (oh_player.py) sends it and answers with an
-- OSD note (已加入處理佇列 / 處理佇列未開啟). Key "a", the last row of the settings menu, `script-message oh-queue-add`.
function proc.add()
    if not active then mp.osd_message("沒有正在播放的影片", 2) return end
    if proc.on then mp.osd_message("已是處理過的影片", 2) return end
    proc.adds = proc.adds + 1
    mp.set_property("user-data/oh/queue-add", tostring(proc.adds))
end

-- ---------------------------------------------------------------------------------------------------------------
-- quality selection
-- ---------------------------------------------------------------------------------------------------------------

local function find_quality(key)
    key = tostring(key)
    for _, q in ipairs(qualities) do
        if q.id == key or q.label == key then return q end
    end
    for _, q in ipairs(qualities) do
        if tostring(q.height) == key then return q end
    end
    return nil
end

-- like a far seek: freeze the current playback, remember the user's pause state, ask Python to reopen the source
-- (another rendition) at the current position and restart Lada there
local function request_quality(key)
    if not active then return end
    if proc.on then mp.osd_message("已處理的影片", 2) return end
    local q = find_quality(key)
    if not q then return end
    if q.id == (quality_wanted or quality_current) then return end
    local pos = display_pos()
    begin_request(true)    -- (also stops a seek tail that is playing)
    counter = counter + 1
    pending = pos
    awaiting = false
    quality_wanted = q.id
    busy_text = "切換畫質 " .. q.label .. "…"
    last_request = string.format("%d:%s", counter, q.id)
    mp.set_property("user-data/oh/quality-pos", string.format("%.3f", pos))
    mp.set_property("user-data/oh/quality-request", last_request)
    show(SHOW_SECS)
end

local menu_keys = { "UP", "DOWN", "ENTER", "ESC", "r" }
local close_smenu                 -- forward declaration (settings menu; opening one menu closes the other)
local close_cmenu                 -- forward declaration (cast menu)

local function close_menu()
    if not menu_open then return end
    menu_open = false
    for _, k in ipairs(menu_keys) do mp.remove_key_binding("oh-menu-" .. k) end
    last_data = nil
    update_click_binding()
    show(SHOW_SECS)
    if draw_info then draw_info() end
end

local function menu_select(i)
    local q = qualities[i]
    close_menu()
    if q then request_quality(q.id) end
end

local function menu_move(d)
    menu_sel = (menu_sel - 1 + d) % #qualities + 1
    request_draw()
end

local function open_menu()
    if menu_open or #qualities < 2 then return end
    close_smenu()
    close_cmenu()
    menu_open = true
    menu_sel = 1
    for i, q in ipairs(qualities) do
        if q.id == (quality_wanted or quality_current) then menu_sel = i end
    end
    mp.add_forced_key_binding("UP", "oh-menu-UP", function() menu_move(-1) end, { repeatable = true })
    mp.add_forced_key_binding("DOWN", "oh-menu-DOWN", function() menu_move(1) end, { repeatable = true })
    mp.add_forced_key_binding("ENTER", "oh-menu-ENTER", function() menu_select(menu_sel) end)
    mp.add_forced_key_binding("ESC", "oh-menu-ESC", close_menu)
    mp.add_forced_key_binding("r", "oh-menu-r", function() menu_move(1) end, { repeatable = true })
    last_data = nil
    update_click_binding()
    show(SHOW_SECS)
    if draw_info then draw_info() end
end

local function toggle_menu()
    if menu_open then close_menu() else open_menu() end
end

local function set_qualities(json)
    local list = {}
    if type(json) == "string" and json ~= "" then
        local v = utils.parse_json(json)
        if type(v) == "table" then list = v end
    end
    qualities = list
    if #qualities < 2 then close_menu() end
    last_data = nil
    if visible() then request_draw() end
end

mp.observe_property("user-data/oh/qualities", "native", function(_, v) set_qualities(v) end)
mp.observe_property("user-data/oh/quality-current", "native", function(_, v)
    quality_current = (v ~= nil) and tostring(v) or nil
    last_data = nil
    if visible() then request_draw() end
end)
mp.register_script_message("oh-quality", request_quality)

-- ---------------------------------------------------------------------------------------------------------------
-- advanced settings
-- ---------------------------------------------------------------------------------------------------------------

local function any_reload_wanted()
    for k in pairs(settings_wanted) do
        local it = setting_by_key(k)
        if it and it.kind == "reload" then return true end
    end
    return false
end

-- choose option `v` of setting `key`: "live" settings are just sent; the others restart the segment at the current
-- position (like a quality switch: playback freezes, the user's pause state is restored once the new one plays)
local function request_setting(key, v)
    local it = setting_by_key(key)
    for _, sub in ipairs(subtitle_settings) do
        if sub.key == key then it = sub break end
    end
    if not it then return end
    v = tostring(v)
    local ok = false
    for _, o in ipairs(it.options or {}) do
        if o.v == v then ok = true end
    end
    if not ok then return end
    if it.kind == "subtitle" then
        if casting then mp.osd_message("字幕設定僅適用於本機播放", 2) return end
        if it.property == "sid" and v ~= "no" then mp.set_property_bool("sub-visibility", true) end
        mp.set_property(it.property, v)
        show(SHOW_SECS)
        return
    end
    if v == setting_shown_value(it) then return end
    if not active and it.kind ~= "live" then return end   -- a restart needs a source
    if it.kind == "live" then
        live_seq = live_seq + 1
        live_wanted[key] = v
        mp.set_property("user-data/oh/settings-set", string.format("%d:%s", live_seq, utils.format_json({ [key] = v })))
        show(SHOW_SECS)
        return
    end
    if proc.on then mp.osd_message("已處理的影片", 2) return end   -- restart / reload kinds need Lada
    if v == it.value then settings_wanted[key] = nil else settings_wanted[key] = v end
    local pos = display_pos()
    begin_request(true)
    counter = counter + 1
    pending = pos
    awaiting = false
    busy_text = any_reload_wanted() and "載入模型中…" or "套用設定…"
    local wanted = {}
    for k, val in pairs(settings_wanted) do wanted[k] = val end
    wanted[key] = v    -- also when going back to the value in use (cancels a pending change)
    last_request = string.format("%d:%s", counter, utils.format_json(wanted))
    mp.set_property("user-data/oh/settings-pos", string.format("%.3f", pos))
    mp.set_property("user-data/oh/settings-request", last_request)
    show(SHOW_SECS)
end

local smenu_keys = { "UP", "DOWN", "ENTER", "ESC", "BS", "LEFT", "RIGHT", "KP_ENTER" }

local function smenu_first_selectable(rows, want)
    for i, r in ipairs(rows) do
        if want(r) then return i end
    end
    for i, r in ipairs(rows) do
        if r.kind ~= "hint" then return i end
    end
    return 1
end

local function smenu_move(d)
    local rows = smenu_rows()
    if #rows == 0 then return end
    local i = smenu_sel
    for _ = 1, #rows do
        i = (i - 1 + d) % #rows + 1
        if rows[i].kind ~= "hint" then break end
    end
    smenu_sel = i
    request_draw()
end

local function smenu_enter_level2(idx)
    local it = menu_settings()[idx]
    if not it then return end
    smenu_level, smenu_item = 2, idx
    local cur = setting_shown_value(it)
    smenu_sel = smenu_first_selectable(smenu_rows(), function(r)
        return r.kind == "opt" and (it.key == "subtitle_delay" or it.options[r.idx].v == cur)
    end)
    last_data = nil
    request_draw()
    draw_info()
end

local function smenu_back()
    if smenu_level == 2 then
        local idx = smenu_item
        smenu_level = 1
        smenu_sel = smenu_first_selectable(smenu_rows(), function(r) return r.idx == idx end)
        last_data = nil
        request_draw()
        draw_info()
    else
        close_smenu()
    end
end

local function smenu_activate(i)
    local rows = smenu_rows()
    local r = rows[i or smenu_sel]
    if not r then return end
    if r.kind == "item" then
        smenu_enter_level2(r.idx)
    elseif r.kind == "action" then
        close_smenu()
        proc.add()
    elseif r.kind == "back" then
        smenu_back()
    elseif r.kind == "opt" then
        local it = menu_settings()[smenu_item]
        if it then request_setting(it.key, it.options[r.idx].v) end
        smenu_back()
    end
end

close_smenu = function()
    if not smenu_open then return end
    smenu_open = false
    for _, k in ipairs(smenu_keys) do mp.remove_key_binding("oh-smenu-" .. k) end
    last_data = nil
    update_click_binding()
    show(SHOW_SECS)
    draw_info()
end

local function open_smenu(subtitles)
    if smenu_open then
        if subtitle_menu == (subtitles == true) then return end
        close_smenu()
    end
    subtitle_menu = subtitles == true
    if #menu_settings() == 0 then return end
    close_menu()
    close_cmenu()
    smenu_open, smenu_level, smenu_sel = true, 1, 1
    mp.add_forced_key_binding("UP", "oh-smenu-UP", function() smenu_move(-1) end, { repeatable = true })
    mp.add_forced_key_binding("DOWN", "oh-smenu-DOWN", function() smenu_move(1) end, { repeatable = true })
    mp.add_forced_key_binding("ENTER", "oh-smenu-ENTER", function() smenu_activate() end)
    mp.add_forced_key_binding("KP_ENTER", "oh-smenu-KP_ENTER", function() smenu_activate() end)
    mp.add_forced_key_binding("RIGHT", "oh-smenu-RIGHT", function()
        local r = smenu_rows()[smenu_sel]
        if r and r.kind == "item" then smenu_activate() end
    end)
    mp.add_forced_key_binding("ESC", "oh-smenu-ESC", smenu_back)
    mp.add_forced_key_binding("BS", "oh-smenu-BS", smenu_back)
    mp.add_forced_key_binding("LEFT", "oh-smenu-LEFT", function() if smenu_level == 2 then smenu_back() end end)
    last_data = nil
    update_click_binding()
    show(SHOW_SECS)
    draw_info()
end

local function toggle_smenu()
    if smenu_open and not subtitle_menu then close_smenu() else open_smenu() end
end

local function toggle_subtitles()
    if smenu_open and subtitle_menu then close_smenu() else open_smenu(true) end
end

local function refresh_subtitles()
    local tracks = {{v = "no", label = "關閉字幕"}}
    for _, track in ipairs(mp.get_property_native("track-list", {}) or {}) do
        if track.type == "sub" then
            local label = track.title or ""
            if label == "" then label = "字幕軌 " .. tostring(track.id) end
            if track.lang and track.lang ~= "" then label = label .. " · " .. track.lang end
            if track.external then label = label .. "（外掛）" end
            tracks[#tracks + 1] = {v = tostring(track.id), label = label}
        end
    end
    local sid = mp.get_property("sid", "no")
    local visible_sub = mp.get_property_bool("sub-visibility", true)
    local scale = mp.get_property_number("sub-scale", 1)
    local delay = mp.get_property_number("sub-delay", 0)
    local sizes = {}
    local scale_label = string.format("%.0f%%", scale * 100)
    local found = false
    for _, size in ipairs({0.5, 0.75, 1, 1.25, 1.5, 1.75, 2}) do
        sizes[#sizes + 1] = {v = tostring(size), label = string.format("%.0f%%", size * 100)}
        if math.abs(scale - size) < 0.001 then found = true end
    end
    if not found then sizes[#sizes + 1] = {v = tostring(scale), label = scale_label} end
    local hint = casting and "僅適用於本機播放；投放端請使用接收裝置設定" or nil
    subtitle_settings = {
        {key = "subtitle_visibility", property = "sub-visibility", kind = "subtitle", label = "字幕顯示",
         hint = hint, value = visible_sub and "yes" or "no",
         options = {{v = "yes", label = "顯示"}, {v = "no", label = "隱藏"}}},
        {key = "subtitle_track", property = "sid", kind = "subtitle", label = "字幕軌道",
         hint = hint or (#tracks == 1 and "此影片沒有可用字幕軌" or "選擇軌道會開啟字幕顯示"), value = sid,
         options = tracks},
        {key = "subtitle_scale", property = "sub-scale", kind = "subtitle", label = "字幕大小",
         hint = hint or "相對原始大小；部分特效字幕不受影響", value = tostring(scale), value_label = scale_label,
         options = sizes},
        {key = "subtitle_delay", property = "sub-delay", kind = "subtitle", label = "字幕時間偏移",
         hint = hint or "正值較晚顯示，負值較早顯示", value = tostring(delay), value_label = string.format("%+.1f 秒", delay),
         options = {{v = tostring(delay - 0.1), label = "提早 0.1 秒"},
                    {v = tostring(delay + 0.1), label = "延後 0.1 秒"}, {v = "0", label = "重設為 0 秒"}}},
    }
    if subtitle_menu and smenu_open then
        local rows = smenu_rows()
        smenu_sel = math.max(1, math.min(smenu_sel, #rows))
        update_click_binding()
    end
    last_data = nil
    if visible() then request_draw() end
end

for _, name in ipairs({"track-list", "sid", "sub-visibility", "sub-scale", "sub-delay"}) do
    mp.observe_property(name, "native", refresh_subtitles)
end
refresh_subtitles()
mp.register_script_message("oh-subtitles-menu", toggle_subtitles)

mp.observe_property("user-data/oh/settings", "native", function(_, v)
    local list = {}
    if type(v) == "string" and v ~= "" then
        local t = utils.parse_json(v)
        if type(t) == "table" then list = t end
    end
    settings = list
    for k, val in pairs(live_wanted) do
        local it = setting_by_key(k)
        if not it or it.value == val then live_wanted[k] = nil end
    end
    if #menu_settings() == 0 then close_smenu() end
    if smenu_level == 2 and not menu_settings()[smenu_item] then smenu_level, smenu_sel = 1, 1 end
    local n = #smenu_rows()
    if smenu_sel > n then smenu_sel = math.max(1, n) end
    last_data = nil
    if visible() then request_draw() end
end)

-- script-message oh-setting <key> <value>   (same path as the menu)
mp.register_script_message("oh-setting", request_setting)
-- script-message oh-settings-menu [open|close|toggle] [key]   (key: open that setting's option list)
mp.register_script_message("oh-settings-menu", function(action, key)
    if action == "close" then close_smenu() return end
    if action == "toggle" or action == nil then
        toggle_smenu()
    else
        open_smenu()
    end
    if smenu_open and key then
        local _, idx = setting_by_key(key)
        if idx then smenu_enter_level2(idx) end
    end
end)

-- ---------------------------------------------------------------------------------------------------------------
-- casting (投放)
-- ---------------------------------------------------------------------------------------------------------------

local cmenu_keys = { "UP", "DOWN", "ENTER", "KP_ENTER", "ESC" }

local function cast_request(id)
    cast_counter = cast_counter + 1
    mp.set_property("user-data/oh/cast-request", string.format("%d:%s", cast_counter, id))
end

local function cmenu_first_item(from, d)
    local n = #cast_rows
    if n == 0 then return 1 end
    local i = from
    for _ = 1, n do
        if cast_rows[i] and cast_rows[i].kind == "item" then return i end
        i = (i - 1 + d) % n + 1
    end
    return from
end

local function cmenu_move(d)
    if #cast_rows == 0 then return end
    cmenu_sel = cmenu_first_item((cmenu_sel - 1 + d) % #cast_rows + 1, d)
    request_draw()
end

close_cmenu = function()
    if not cmenu_open then return end
    cmenu_open = false
    for _, k in ipairs(cmenu_keys) do mp.remove_key_binding("oh-cmenu-" .. k) end
    last_data = nil
    update_click_binding()
    show(SHOW_SECS)
    if draw_info then draw_info() end
end

local function cmenu_activate(i)
    local r = cast_rows[i or cmenu_sel]
    if not r or r.kind ~= "item" or not r.id or r.id == "" then return end
    if not r.id:match("^pin:") and not r.id:match("^pair:") then close_cmenu() end
    cast_request(r.id)
end

local function open_cmenu()
    if cmenu_open or not active then return end
    close_menu()
    close_smenu()
    cmenu_open = true
    cmenu_sel = cmenu_first_item(1, 1)
    cast_request("menu")          -- Python rescans if its device list is stale and republishes the rows
    mp.add_forced_key_binding("UP", "oh-cmenu-UP", function() cmenu_move(-1) end, { repeatable = true })
    mp.add_forced_key_binding("DOWN", "oh-cmenu-DOWN", function() cmenu_move(1) end, { repeatable = true })
    mp.add_forced_key_binding("ENTER", "oh-cmenu-ENTER", function() cmenu_activate() end)
    mp.add_forced_key_binding("KP_ENTER", "oh-cmenu-KP_ENTER", function() cmenu_activate() end)
    mp.add_forced_key_binding("ESC", "oh-cmenu-ESC", close_cmenu)
    last_data = nil
    update_click_binding()
    show(SHOW_SECS)
    if draw_info then draw_info() end
end

local function toggle_cmenu()
    if cmenu_open then close_cmenu() else open_cmenu() end
end

mp.observe_property("user-data/oh/cast-menu", "native", function(_, v)
    local list = {}
    if type(v) == "string" and v ~= "" then
        local t = utils.parse_json(v)
        if type(t) == "table" then list = t end
    end
    cast_rows = list
    if cmenu_sel > #cast_rows or not (cast_rows[cmenu_sel] and cast_rows[cmenu_sel].kind == "item") then
        cmenu_sel = cmenu_first_item(1, 1)
    end
    last_data = nil
    if visible() then request_draw() end
    if cmenu_open and draw_info then draw_info() end
end)

mp.observe_property("user-data/oh/casting", "native", function(_, v)
    casting = (v == "1")
    refresh_subtitles()
    last_data = nil
    show(SHOW_SECS)
end)

mp.observe_property("user-data/oh/cast-position", "number", function()
    if casting then request_draw() end
end)

mp.observe_property("user-data/oh/cast-paused", "native", function()
    if casting then
        last_data = nil
        show(SHOW_SECS)
    end
end)

-- a quality / settings request Python applied without loading a local segment (while casting)
mp.observe_property("user-data/oh/request-done", "native", function(_, v)
    if type(v) ~= "string" or v == "" then return end
    if pending and v == last_request then
        pending, awaiting = nil, false
        quality_wanted, settings_wanted = nil, {}
        last_data = nil
        show(SHOW_SECS)
    end
end)

-- script-message oh-restart <pos> <paused 0|1>: restart the local segment at pos through Python (casting ended)
mp.register_script_message("oh-restart", function(t, paused)
    t = tonumber(t)
    if not t or not active then return end
    local dur = duration()
    if dur > 0 then t = math.min(t, dur - 1) end
    t = math.max(0, t)
    user_paused = (paused == "1")
    mp.set_property_bool("pause", true)
    counter = counter + 1
    pending = t
    awaiting = false
    busy_text = "處理中…"
    last_request = string.format("%d:%.3f", counter, t)
    mp.set_property("user-data/oh/seek-request", last_request)
    show(SHOW_SECS)
end)

-- script-message oh-cast-menu [open|close|toggle]
mp.register_script_message("oh-cast-menu", function(action)
    if action == "close" then close_cmenu()
    elseif action == "open" then open_cmenu()
    else toggle_cmenu() end
end)

-- ---------------------------------------------------------------------------------------------------------------
-- info panel (進階資訊)
-- ---------------------------------------------------------------------------------------------------------------

local LEVEL_COLOR = { good = "64DC50", warn = "40D0FF", bad = "5A5AFF", idle = "A0A0A0" }   -- BGR
local CHART_LEN = 120      -- samples a chart spans (one per second, as info_panel.HISTORY_LEN); newest at the right
local CHART_COLOR = "FFC864"

-- trend chart of row r in the box x0, y0, w x h: faint backdrop, filled area + line per run of values, reference line
local function chart(x0, y0, w, h, r, bgr, lw)
    local out = { rect(x0, y0, x0 + w, y0 + h, "FFFFFF", "E6") }
    local v, n = r.h, #r.h
    local hmax = tonumber(r.hmax) or 1
    if hmax <= 0 then hmax = 1 end
    local dx = w / (math.max(CHART_LEN, n) - 1)
    local function X(i) return x0 + w - (n - i) * dx end
    local function Y(val) return y0 + h - h * math.max(0, math.min(1, val / hmax)) end
    local function ok(i) return type(v[i]) == "number" and v[i] >= 0 end
    local i = 1
    while i <= n do
        while i <= n and not ok(i) do i = i + 1 end
        if i > n then break end
        local j = i
        while j < n and ok(j + 1) do j = j + 1 end
        if j > i then
            local area, line = { { X(i), y0 + h } }, {}
            for k = i, j do
                area[#area + 1] = { X(k), Y(v[k]) }
                line[#line + 1] = { X(k), Y(v[k]) - lw / 2 }
            end
            area[#area + 1] = { X(j), y0 + h }
            for k = j, i, -1 do line[#line + 1] = { X(k), Y(v[k]) + lw / 2 } end
            out[#out + 1] = poly(area, bgr, "B0")
            out[#out + 1] = poly(line, bgr, "10")
        else
            out[#out + 1] = rect(X(i) - lw, Y(v[i]) - lw, X(i) + lw, Y(v[i]) + lw, bgr, "10")
        end
        i = j + 1
    end
    if tonumber(r.href) and r.href < hmax then
        local yr = Y(r.href)
        out[#out + 1] = rect(x0, yr - lw / 3, x0 + w, yr + lw / 3, "FFFFFF", "70")
    end
    return table.concat(out, "\n")
end

draw_info = function()
    if not info_visible then
        info_overlay:remove()
        return
    end
    local g = geom()
    if not g then return end
    local u = g.u
    local sections = (info_data and info_data.sections) or { { title = "進階資訊", rows = { { k = "狀態", v = "讀取中…" } } } }

    -- free area: top-left, above the control bar and the space a hover thumbnail can take, clear of open menus
    local x0, y0 = 12 * u, 12 * u
    local tw = math.max(120, math.min((thumb and thumb.w or 320) * 1.75, g.w * THUMB_FRAC))
    local aspect = (thumb and thumb.w > 0) and (thumb.h / thumb.w) or (9 / 16)
    local bottom = math.min(g.h - 92 * u, g.h - 66 * u - 20 * u - tw * aspect) - 8 * u
    local right = g.w - 12 * u
    local menus = {}
    if menu_open and g.m_x0 then menus[#menus + 1] = { g.m_x0, g.m_y0 } end
    if smenu_open and g.s_x0 then menus[#menus + 1] = { g.s_x0, g.s_y0 } end
    if cmenu_open and g.c_x0 then menus[#menus + 1] = { g.c_x0, g.c_y0 } end
    for _, m in ipairs(menus) do
        if m[1] - 10 * u - x0 >= 220 * u then
            right = math.min(right, m[1] - 10 * u)      -- room on the left of the menu: stay left of it
        else
            bottom = math.min(bottom, m[2] - 10 * u)    -- otherwise stay above it
        end
    end
    local avail_w, avail_h = right - x0, bottom - y0
    if avail_w < 60 or avail_h < 30 then
        info_overlay:remove()
        return
    end

    -- one column of sections: width / height at font size fs (with charts: room for them right of the values)
    local function measure(secs, fs, charts)
        local pad, lh, hh, gap = 0.8 * fs, 1.36 * fs, 1.5 * fs, 0.4 * fs
        local kw, vw, n_rows, tw_head, has_chart = 0, 0, 0, 0, false
        for _, s in ipairs(secs) do
            tw_head = math.max(tw_head, text_w(s.title or "", fs * 1.05))
            for _, r in ipairs(s.rows or {}) do
                kw = math.max(kw, text_w(r.k or "", fs * 0.92))
                vw = math.max(vw, text_w(r.v or "", fs))
                n_rows = n_rows + 1
                if type(r.h) == "table" and #r.h > 0 then has_chart = true end
            end
        end
        local cw = (charts and has_chart) and 7 * fs or 0
        local width = math.max(tw_head, kw + 1.1 * fs + vw + (cw > 0 and (1.0 * fs + cw) or 0))
        local height = #secs * hh + n_rows * lh + math.max(0, #secs - 1) * gap
        return width, height, { pad = pad, lh = lh, hh = hh, gap = gap, kw = kw, vw = vw, cw = cw }
    end

    -- split into ncols columns (in order, balancing the row count); returns {cols, width, height} at fs
    local function layout(secs, ncols, fs, charts)
        local cols = { secs }
        if ncols == 2 and #secs >= 2 then
            local total = 0
            for _, s in ipairs(secs) do total = total + #(s.rows or {}) + 1.5 end
            local best, best_k, acc = math.huge, 1, 0
            for k = 1, #secs - 1 do
                acc = acc + #(secs[k].rows or {}) + 1.5
                local m = math.max(acc, total - acc)
                if m < best then best, best_k = m, k end
            end
            local a, b = {}, {}
            for i, s in ipairs(secs) do
                if i <= best_k then a[#a + 1] = s else b[#b + 1] = s end
            end
            cols = { a, b }
        end
        local pad = 0.8 * fs
        local width, height, out = 2 * pad + (#cols - 1) * 1.4 * fs, 0, {}
        for _, c in ipairs(cols) do
            local cw, ch, m = measure(c, fs, charts)
            out[#out + 1] = { secs = c, w = cw, m = m }
            width = width + cw
            height = math.max(height, ch)
        end
        return { cols = out, w = width, h = height + 2 * pad, pad = pad }
    end

    -- the most content that still fits at a readable size: one or two columns; on small windows sections are
    -- dropped, least important first (影片, then 設定, then GPU, then 處理狀態; 投放 stays)
    local drop_order = { ["影片"] = 4, ["設定"] = 3, ["GPU"] = 2, ["投放"] = 0 }
    local base = 13 * u
    local secs, chosen = sections, nil
    while not chosen do
        local best = nil
        for ncols = 1, 2 do
            local l = layout(secs, ncols, base)
            local scale = math.min(1, avail_w / l.w, avail_h / l.h)
            -- a second column only when one column would have to shrink noticeably
            if not best or (best.fs < 0.8 * base and base * scale > best.fs + 0.5) then
                best = { ncols = ncols, fs = base * scale }
            end
        end
        if best.fs >= 10 or #secs <= 1 then
            chosen = best
            break
        end
        local worst, wi = 0, nil
        for i, s in ipairs(secs) do
            local d = drop_order[s.title] or 1
            if d > worst then worst, wi = d, i end
        end
        local trimmed = {}
        for i, s in ipairs(secs) do
            if i ~= wi then trimmed[#trimmed + 1] = s end
        end
        secs = trimmed
    end
    local fs = math.max(chosen.fs, 8)
    -- trend charts: they only add width, so whenever there is room beside the values at the size the panel is
    -- drawn anyway (and that size is still comfortably readable)
    local charts = fs >= 10 * u and layout(secs, chosen.ncols, fs, true).w <= avail_w
    local l = layout(secs, chosen.ncols, fs, charts)
    local bw, bh = math.min(l.w, avail_w), math.min(l.h, avail_h)

    local p = {}
    local function add(s) p[#p + 1] = s end
    add(rrect(x0, y0, x0 + bw, y0 + bh, 8 * u, "141414", "58"))
    local cx = x0 + l.pad
    for _, col in ipairs(l.cols) do
        local m = col.m
        local y = y0 + l.pad
        local kx, vx = cx, cx + m.kw + 1.1 * fs
        for si, s in ipairs(col.secs) do
            if si > 1 then
                add(rect(cx, y + m.gap / 2 - 0.5 * u, cx + col.w, y + m.gap / 2 + 0.5 * u, "FFFFFF", "C8"))
                y = y + m.gap
            end
            add(ptext(kx, y + m.hh / 2, 4, fs * 1.05, ass_esc(s.title), "00B4FF", true))
            y = y + m.hh
            for _, r in ipairs(s.rows or {}) do
                if y + m.lh > y0 + bh then break end
                add(ptext(kx, y + m.lh / 2, 4, fs * 0.92, ass_esc(r.k), "B4B4B4"))
                add(ptext(vx, y + m.lh / 2, 4, fs, ass_esc(r.v), LEVEL_COLOR[r.c or ""] or "FFFFFF"))
                if m.cw > 0 and type(r.h) == "table" and #r.h > 0 then
                    add(chart(vx + m.vw + 1.0 * fs, y + 0.16 * m.lh, m.cw, 0.68 * m.lh, r,
                        LEVEL_COLOR[r.c or ""] or CHART_COLOR, math.max(1, 1.4 * u)))
                end
                y = y + m.lh
            end
        end
        cx = cx + col.w + 1.4 * fs
    end
    info_overlay.res_x, info_overlay.res_y = g.w, g.h
    info_overlay.data = table.concat(p, "\n")
    info_overlay:update()
end

-- mpv's own OSD messages (volume, "載入模型中…") sit top-left by default: move them to the top centre while the
-- panel occupies that corner
local osd_align_orig = mp.get_property("osd-align-x") or "left"

local function set_info_visible(on)
    info_visible = on and true or false
    mp.set_property("user-data/oh/info-visible", info_visible and "1" or "")
    mp.set_property("osd-align-x", info_visible and "center" or osd_align_orig)
    draw_info()
    if visible() then request_draw() end
end

local function toggle_info()
    set_info_visible(not info_visible)
end

mp.set_property("user-data/oh/info-visible", "")
mp.observe_property("user-data/oh/progress", "native", function(_, v)
    progress = nil
    if type(v) == "string" and v ~= "" then
        local t = utils.parse_json(v)
        if type(t) == "table" then progress = t end
    end
    if visible() then request_draw(true) end
end)

mp.observe_property("user-data/oh/info", "native", function(_, v)
    if type(v) ~= "string" or v == "" then return end
    local t = utils.parse_json(v)
    if type(t) == "table" then
        info_data = t
        draw_info()
    end
end)
-- script-message oh-info [on|off|toggle]
mp.register_script_message("oh-info", function(action)
    if action == "on" then set_info_visible(true)
    elseif action == "off" then set_info_visible(false)
    else toggle_info() end
end)

-- Automatic source switches use an independent token handshake. User seeks keep served / resumed below.
proc.handoff.read = function()
    local raw = mp.get_property_native("user-data/oh/handoff")
    if type(raw) ~= "string" or raw == "" then return nil end
    local ok, h = pcall(utils.parse_json, raw)
    if not ok or type(h) ~= "table" or type(h.token) ~= "string" or h.token == ""
        or type(h.url) ~= "string" or h.url == "" or type(h.paused) ~= "boolean" then return nil end
    local position = tonumber(h.position)
    if not position then return nil end
    return { token = h.token, url = h.url, paused = h.paused, position = position }
end

proc.handoff.finish = function()
    local h = proc.handoff.pending_ack
    if not h then return end
    if pending or mp.get_property("path", "") ~= h.url then
        proc.handoff.pending_ack = nil
        return
    end
    local ready = h.paused
    if not ready then
        local pos = mp.get_property_number("time-pos")
        ready = mp.get_property_native("core-idle") == false
            and not mp.get_property_bool("seeking", false) and pos ~= nil and pos >= h.position
    end
    if ready then
        proc.handoff.pending_ack = nil
        if proc.handoff.last_ack_token ~= h.token then
            proc.handoff.last_ack_token = h.token
            mp.set_property("user-data/oh/handoff-resumed", h.token)
        end
    end
end

proc.handoff.cancel = function()
    proc.handoff.pending_ack = nil
    -- Clearing the published request also tells the controller that a newer user action superseded it.
    mp.set_property("user-data/oh/handoff", "")
end

mp.register_event("file-loaded", function()
    local expect = mp.get_property_native("user-data/oh/expect-load")
    local actual_path = mp.get_property("path", "")
    internal_active = type(expect) == "string" and expect ~= "" and actual_path == expect
    -- (get_property would return the user-data string JSON-quoted)
    local served = mp.get_property_native("user-data/oh/served")
    local matched = false
    local handoff = proc.handoff.read()
    local handoff_path_match = handoff and handoff.url == actual_path
    if handoff_path_match and handoff.token ~= proc.handoff.last_ack_token then
        -- Even while a newer user seek is pending, an old handoff must not borrow its served value.
        if not pending then
            proc.handoff.pending_ack = handoff
            mp.set_property_bool("pause", handoff.paused)
            proc.handoff.finish()
        end
    elseif pending and served == last_request then
        await_target, pending, matched = pending, nil, true
    elseif initial_loading and not pending then
        await_target, matched = start_pos(), true
    end
    initial_loading = false
    if matched then
        quality_wanted = nil
        settings_wanted = {}   -- Python has republished the settings in use by now
        awaiting, await_served = true, served
        mp.set_property_bool("pause", user_paused)
        if user_paused then finish_await() end
        user_paused = false
    end
    show(SHOW_SECS)
end)

mp.observe_property("core-idle", "bool", function(_, idle)
    if awaiting and idle == false and mp.get_property_number("time-pos") then finish_await() end
    if idle == false then proc.handoff.finish() end
end)

mp.observe_property("pause", "bool", function(_, paused)
    if awaiting and paused then finish_await() end
    show(SHOW_SECS)
end)

mp.observe_property("osd-dimensions", "native", function()
    last_data = nil
    if visible() then request_draw() end
    if draw_info then draw_info() end
end)

-- ---------------------------------------------------------------------------------------------------------------
-- mouse: show on movement, hover tooltip, click on the bar seeks
-- ---------------------------------------------------------------------------------------------------------------

-- While a restart is pending the real pause property is ours (it freezes the old segment); SPACE then flips the
-- state we restore once the new segment is loaded. While a seek tail plays the old timeline (not frozen), SPACE
-- pauses / resumes that too.
local function toggle_pause()
    if not active then return end
    if casting then
        counter = counter + 1
        mp.set_property("user-data/oh/cast-request", tostring(counter) .. ":play_pause")
        show(SHOW_SECS)
        return
    end
    if pending then
        user_paused = not user_paused
        if tail_playing then mp.set_property_bool("pause", user_paused) end
        show(SHOW_SECS)
    else
        local handoff = proc.handoff.read()
        if handoff and handoff.token ~= proc.handoff.last_ack_token then
            -- A user's pause change during a load updates the handoff intent, so file-loaded cannot undo it.
            handoff.paused = not mp.get_property_bool("pause", false)
            mp.set_property("user-data/oh/handoff", utils.format_json(handoff))
            if proc.handoff.pending_ack then proc.handoff.pending_ack.paused = handoff.paused end
            mp.set_property_bool("pause", handoff.paused)
            proc.handoff.finish()
        else
            mp.commandv("cycle", "pause")
        end
    end
end
local function toggle_fullscreen()
    mp.commandv("cycle", "fullscreen")
end

-- volume: 0-100 %, OSD "音量 NN%" / "靜音"
-- drawn by draw() above the speaker button (mpv's own OSD corner is where the info panel lives)
local function volume_osd()
    if mp.get_property_bool("mute", false) then
        vol_osd_text = "靜音"
    else
        vol_osd_text = string.format("音量 %d%%", math.floor((mp.get_property_number("volume", 0) or 0) + 0.5))
    end
    vol_osd_until = mp.get_time() + 1.5
end

local function set_volume(v)
    v = math.max(0, math.min(100, math.floor(v + 0.5)))
    mp.set_property_number("volume", v)
    if v > 0 and mp.get_property_bool("mute", false) then mp.set_property_bool("mute", false) end
    volume_osd()
    show(SHOW_SECS)
end

local function change_volume(d)
    set_volume((mp.get_property_number("volume", 100) or 100) + d)
end

local function toggle_mute()
    mp.set_property_bool("mute", not mp.get_property_bool("mute", false))
    volume_osd()
    show(SHOW_SECS)
end

local function volume_from_x(g, x)
    if not g or not g.vol_x0 then return end
    set_volume((x - g.vol_x0) / (g.vol_x1 - g.vol_x0) * 100)
end

mp.observe_property("volume", "number", function() if visible() then request_draw(true) end end)
mp.observe_property("mute", "bool", function() if visible() then request_draw(true) end end)

local function on_click(held)    -- held: the button is down (a release will follow), so the slider can be dragged
    local g = geom()
    local what = hit(g)
    if menu_open and what ~= "menu" and what ~= "quality" then
        close_menu()      -- a click anywhere else only dismisses the menu (the other menu buttons switch menus)
        if what ~= "gear" and what ~= "cast" and what ~= "subtitles" then return end
    end
    if smenu_open and what ~= "smenu" and what ~= "gear" and what ~= "subtitles" then
        close_smenu()
        if what ~= "quality" and what ~= "cast" then return end
    end
    if cmenu_open and what ~= "cmenu" and what ~= "cast" then
        close_cmenu()
        if what ~= "quality" and what ~= "gear" and what ~= "subtitles" then return end
    end
    if what == "subtitles" then
        toggle_subtitles()
    elseif what == "cast" then
        toggle_cmenu()
    elseif what == "cmenu" then
        local i = cmenu_row_at(g)
        if i and i > 0 then
            cmenu_sel = i
            cmenu_activate(i)
        end
    elseif what == "spk" then
        toggle_mute()
    elseif what == "vol" then
        vol_drag = held and true or false
        volume_from_x(g, mouse.x)
    elseif what == "gear" then
        toggle_smenu()
    elseif what == "info" then
        toggle_info()
    elseif what == "smenu" then
        local i = smenu_row_at(g)
        if i and i > 0 then
            smenu_sel = i
            smenu_activate(i)
        end
    elseif what == "play" then
        toggle_pause()
    elseif what == "fs" then
        toggle_fullscreen()
    elseif what == "quality" then
        toggle_menu()
    elseif what == "menu" then
        local i = menu_item_at(g)
        if i and i > 0 then menu_select(i) end
    elseif what == "bar" then
        seek_to(x_to_time(g, mouse.x))
    end
end

-- The click bindings exist only while the cursor is over the bar or a button (and are dropped as soon as it leaves
-- them, leaves the window, or the bar hides), so everywhere else mpv's default handling (window dragging,
-- double-click fullscreen) is untouched.
update_click_binding = function()
    local want = hit() ~= nil or menu_open or smenu_open or cmenu_open or vol_drag
    if want and not click_bound then
        -- complex: the press acts (as before), the release ends a volume-slider drag
        mp.add_forced_key_binding("MBTN_LEFT", "oh-click", function(e)
            if e.event == "down" or e.event == "press" then   -- "press": synthetic clicks (mouse command)
                on_click(e.event == "down")
            elseif e.event == "up" and vol_drag then
                vol_drag = false
                update_click_binding()
                show(SHOW_SECS)
            end
        end, { complex = true })
        -- the second click of a double click must not repeat the action (menu rows / menu & toggle buttons / dismissal)
        mp.add_forced_key_binding("MBTN_LEFT_DBL", "oh-click-dbl", function()
            local what = hit()
            if menu_open or smenu_open or cmenu_open or what == "menu" or what == "quality" or what == "smenu"
                or what == "gear" or what == "subtitles" or what == "info" or what == "spk" or what == "vol" or what == "cast"
                or what == "cmenu" then return end
            on_click()
        end)
        click_bound = true
    elseif not want and click_bound then
        mp.remove_key_binding("oh-click")
        mp.remove_key_binding("oh-click-dbl")
        click_bound = false
    end
end

mp.observe_property("mouse-pos", "native", function(_, v)
    mouse = v or { x = -1, y = -1, hover = false }
    local xy = tostring(mouse.x) .. "," .. tostring(mouse.y)
    if menu_open then
        local i = menu_item_at(geom() or {})
        if i and i > 0 then menu_sel = i end
    end
    if vol_drag then volume_from_x(geom(), mouse.x) end
    if smenu_open then
        local i = smenu_row_at(geom() or {})
        if i and i > 0 then smenu_sel = i end
    end
    if cmenu_open then
        local i = cmenu_row_at(geom() or {})
        if i and i > 0 then cmenu_sel = i end
    end
    update_click_binding()
    if not in_zone() then hide_thumb(true) end
    if mouse.hover ~= false and xy ~= last_mouse_xy then
        last_mouse_xy = xy
        show_until = math.max(show_until, mp.get_time() + SHOW_SECS)
        request_draw(true)
    elseif hit() ~= nil then
        request_draw(true)
    end
end)

-- Python's answer to thumb-request
mp.observe_property("user-data/oh/thumb-ready", "native", function(_, v)
    if type(v) ~= "string" then return end
    local req, path, w, h = v:match("^(%-?%d+)|(.-)|(%d+)|(%d+)|%d|%d+$")
    if not path then return end
    thumb = { path = path, w = tonumber(w), h = tonumber(h), req = tonumber(req) }
    if in_zone() then request_draw(true) end
end)

mp.observe_property("fullscreen", "bool", function()
    last_data = nil
    if visible() then request_draw() end
end)

-- ---------------------------------------------------------------------------------------------------------------
-- persistent player: status text, dropped loads are discarded, per-source reset
-- ---------------------------------------------------------------------------------------------------------------

local status_overlay = mp.create_osd_overlay("ass-events")
status_overlay.z = 5

-- centred text: first line large, further lines smaller (e.g. the idle hint + "模型載入中…")
local function draw_status()
    local s = status_text
    if s == nil and idle_mode then s = IDLE_HINT end
    if type(s) ~= "string" or s == "" then
        status_overlay:remove()
        return
    end
    local w, h = mp.get_osd_size()
    if not w or w <= 0 or h <= 0 then return end
    local u = ui_scale()
    local lines = {}
    for ln in s:gmatch("[^\n]+") do lines[#lines + 1] = ln end
    local fs1, fs2 = 30 * u, 17 * u
    local bw, bh = 0, 0
    for i, ln in ipairs(lines) do
        local fs = i == 1 and fs1 or fs2
        bw = math.max(bw, text_w(ln, fs))
        bh = bh + fs * (i == 1 and 1.45 or 1.5)
    end
    bw = math.min(w - 16 * u, bw + 56 * u)
    bh = bh + 30 * u
    local x0, y0 = (w - bw) / 2, (h - bh) / 2
    local p = { rrect(x0, y0, x0 + bw, y0 + bh, 10 * u, "141414", "60") }
    local y = y0 + 15 * u
    for i, ln in ipairs(lines) do
        local fs = i == 1 and fs1 or fs2
        local lh = fs * (i == 1 and 1.45 or 1.5)
        p[#p + 1] = ptext(w / 2, y + lh / 2, 5, fs, ass_esc(ln), i == 1 and "FFFFFF" or "B4B4B4", i == 1)
        y = y + lh
    end
    status_overlay.res_x, status_overlay.res_y = w, h
    status_overlay.data = table.concat(p, "\n")
    status_overlay:update()
end

mp.observe_property("user-data/oh/status", "native", function(_, v)
    if type(v) == "string" then status_text = v end
    draw_status()
end)
mp.observe_property("osd-dimensions", "native", function() draw_status() end)

-- user-data/oh/processed = "1" while a queue output plays as is (Python publishes it before the load)
mp.observe_property("user-data/oh/processed", "native", function(_, v)
    proc.on = (v == "1")
    last_data = nil
    if visible() then request_draw() end
end)

-- The controller temporarily plays the source while the queue repairs ahead. Keep this as player state only;
-- processed mode stays off so the normal progressive seek rules remain active.
mp.observe_property("user-data/oh/passthrough", "native", function(_, v)
    local on = v == "1"
    if on ~= proc.handoff.passthrough then
        proc.handoff.passthrough = on
        if on then mp.osd_message("原片播放中，背景修復中", 2) end
    end
end)

-- user-data/oh/ranges (see header): parse only when the string changes
set_ranges = function(raw)
    raw = type(raw) == "string" and raw or ""
    if raw == proc.ranges_raw then return end
    proc.ranges_raw = raw
    local rs = {}
    for a, b in raw:gmatch("([%d%.]+)%-([%d%.]+)") do
        a, b = tonumber(a), tonumber(b)
        if a and b and b >= a then rs[#rs + 1] = { a, b } end
    end
    proc.ranges = rs
    last_data = nil
    if visible() then request_draw() end
end
mp.observe_property("user-data/oh/ranges", "native", function(_, v) set_ranges(v) end)

mp.observe_property("user-data/oh/active", "native", function(_, v)
    if v == nil then return end
    local was = active
    active = (v == "1")
    if active ~= was then
        last_data = nil
        if active then show(SHOW_SECS) else overlay:remove() hide_thumb(true) update_click_binding() end
    end
    if not active and (#proc.ranges > 0 or proc.ranges_raw ~= "") then
        set_ranges("")
        mp.set_property("user-data/oh/ranges", "")   -- so that an identical value published later is seen as a change
    end
end)

-- intercepted loads arrive one by one (a multi-file drop is several loadfile commands): one OSD message for all
local function queue_drop(item)
    if type(item) ~= "string" or item == "" then return end
    drop_items[#drop_items + 1] = item
    if drop_timer then drop_timer:kill() end
    drop_timer = mp.add_timeout(0.2, function()
        drop_timer = nil
        local items = drop_items
        drop_items = {}
        if #items > 0 then mp.osd_message("請從主程式開啟影片", 3) end
    end)
end

local function is_internal(url)
    if type(url) ~= "string" or url == "" then return false end
    local expect = mp.get_property_native("user-data/oh/expect-load")
    if type(expect) == "string" and expect ~= "" and url == expect then return true end
    local raw = mp.get_property_native("user-data/oh/internal-urls")
    if type(raw) ~= "string" or raw == "" then return false end
    local ok, urls = pcall(utils.parse_json, raw)
    if not ok or type(urls) ~= "table" then return false end
    for _, owned in ipairs(urls) do
        if type(owned) == "string" and owned == url then return true end
    end
    return false
end

-- one of Python's segment streams (the current one, or an old one left in the playlist after a seek tail handed over)
local function is_segment(url)
    return type(url) == "string" and url:match("^tcp://127%.0%.0%.1:%d+$") ~= nil
end

-- Playlist entries that are not playing: the expected append stays; stale controller-owned URLs and feeder segments
-- are removed silently, while user drops are taken out and discarded.
local function clean_playlist()
    local pl = mp.get_property_native("playlist") or {}
    local found = {}
    for i = #pl, 1, -1 do
        local e = pl[i]
        local expect = mp.get_property_native("user-data/oh/expect-load")
        local is_expected = type(expect) == "string" and expect ~= "" and e.filename == expect
        if not e.current and not e.playing and not is_expected then
            mp.commandv("playlist-remove", tostring(i - 1))
            -- Controller-owned source / feeder URLs can be stale playlist entries; discard them silently.
            if not is_internal(e.filename) and not is_segment(e.filename) then table.insert(found, 1, e.filename) end
        end
    end
    for _, f in ipairs(found) do queue_drop(f) end
end

-- Every load that is not one of Python's segments (drag & drop while idle, `loadfile <x>` from anywhere) is
-- cancelled before mpv opens anything and discarded (OSD hint). Runs before every other on_load hook.
mp.add_hook("on_load", 5, function()
    local url = mp.get_property("stream-open-filename", "")
    if is_internal(url) then return end
    local resume = internal_active and active and last_pos
    internal_active = false
    mp.commandv("stop")       -- the load is aborted once the hook returns
    msg.info("intercepted load: " .. url)
    queue_drop(url)
    if resume then
        -- a "replace" load already ended the segment that was playing: bring it back (Python restarts it there)
        mp.add_timeout(0.05, function() seek_to(resume) end)
    end
end)

-- --drag-and-drop=append adds drops behind the playing segment: take them out again and discard them
mp.observe_property("playlist-count", "number", function(_, n)
    if not n or n < 1 then return end
    clean_playlist()
end)
-- after a seek tail handed over (the new segment was appended), the old segment is left in front of it: drop it
mp.register_event("file-loaded", function()
    if (mp.get_property_number("playlist-count", 1) or 1) > 1 then clean_playlist() end
end)

-- user-data/oh/tail = "1" while Python keeps the old timeline playing for a pending seek (see header)
mp.observe_property("user-data/oh/tail", "native", function(_, v)
    tail_playing = (v == "1")
end)

mp.observe_property("time-pos", "number", function(_, v)
    if v and internal_active then last_pos = v end
    proc.handoff.finish()
end)
mp.observe_property("idle-active", "bool", function(_, idle)
    if idle then internal_active = false end
end)

-- per-source state is cleared before Python starts another source
mp.register_script_message("oh-reset", function(tag)
    -- every step is isolated: one failing step must neither leave the request properties uncleared nor withhold the
    -- ack (a stale seek-request would be replayed on the new source: it would not start at its beginning)
    local function step(f, ...)
        local ok, err = pcall(f, ...)
        if not ok then msg.error("oh-reset: " .. tostring(err)) end
    end
    if proc.reset_compare then step(proc.reset_compare) end
    counter = 0
    last_request, pending, awaiting, await_target, await_served = nil, nil, false, nil, nil
    initial_loading, user_paused, tail_playing = true, false, false
    proc.handoff.passthrough, proc.handoff.pending_ack, proc.handoff.last_ack_token = false, nil, nil
    proc.handoff.compare_passthrough = false
    progress = nil
    step(set_ranges, "")
    quality_wanted, settings_wanted, live_wanted = nil, {}, {}
    for _, k in ipairs({ "seek-request", "quality-request", "settings-request", "quality-pos", "settings-pos",
        "resumed", "served", "ranges", "thumb-ready", "thumb-request", "duration", "start", "qualities", "quality-current",
        "handoff", "handoff-resumed", "internal-urls", "passthrough" }) do
        step(mp.set_property, "user-data/oh/" .. k, "")
    end
    step(close_menu)
    step(close_smenu)
    step(close_cmenu)
    qualities, quality_current = {}, nil
    step(hide_thumb, true)
    thumb = nil
    internal_active, last_pos = false, nil
    thumb_req = ""
    active = false
    step(mp.set_property, "user-data/oh/active", "0")   -- Python sets "1" once the new source is published
    info_data = nil
    step(draw_info)
    step(function() overlay:remove() end)
    last_data = nil
    step(update_click_binding)
    mp.set_property("user-data/oh/reset-ack", tostring(tag or ""))
end)

-- ---------------------------------------------------------------------------------------------------------------
-- RTX Video Super Resolution (see header)
-- ---------------------------------------------------------------------------------------------------------------

local VSR_LABEL = "oh-vsr"
local vsr_on = mp.get_opt("oh-vsr") == "on"             -- startup value; then user-data/oh/vsr
local vsr_screen_h = tonumber(mp.get_opt("oh-screen-h") or "")   -- primary monitor height (Python), a fallback
local vsr_applied = nil           -- filter string we added (nil = none)
local vsr_failed = nil            -- filter string mpv refused: not retried until the wanted filter changes
local vsr_warned = false          -- the OSD message about a failure was shown
local vsr_status_shown = nil      -- last user-data/oh/vsr-status we set
local cmp = { vsr_off = false }  -- compare state (one local: the chunk is near Lua's 200 locals limit); stages (below) switch VSR off temporarily; the user's setting stays in vsr_on

local function vsr_remove(force)
    if vsr_applied or force then
        mp.commandv("vf", "remove", "@" .. VSR_LABEL)   -- no error when the label is not there
        vsr_applied = nil
    end
end

-- (re)decide the filter: called whenever the setting, the video height, the display or the decoder changes. Every
-- vf change reinitialises the filter chain (a visible stutter), so an unchanged filter string is never re-added.
local function vsr_update()
    local status
    if lada_platform == "darwin" then
        vsr_remove()
        status = "不可用（需要 Windows／RTX）"
    elseif cmp.vsr_off then
        vsr_remove()      -- temporary compare override: keep the published status of the real setting
    elseif not vsr_on then
        vsr_remove()
        vsr_failed = nil
        status = "關"
    else
        local hw = mp.get_property("hwdec-current", "")
        local h = mp.get_property_number("video-params/h")
        -- compare graph: the decoder belongs to the graph (hwdec-current is empty) and mpv uploads its software frames
        -- to d3d11 for d3d11vpp itself, so any decoder will do
        if cmp.graph then hw = "d3d11va" end
        if hw == "" or not h or h <= 0 then
            -- no video (between segments, idle): leave the filter as it is, the next segment is the same video
            if not vsr_applied then status = "開（等待影片）" end
        elseif not hw:find("^d3d11va") then
            vsr_remove()      -- d3d11vpp needs a d3d11va decoder; "-copy" is fine (mpv uploads the frames again, compare graph)
            status = "不可用（非 d3d11va 硬解）"
        else
            local dh = mp.get_property_number("display-height") or vsr_screen_h
            local scale = dh and dh > 0 and math.min(4, dh / h) or nil
            if not scale then
                status = "開（螢幕解析度未知）"
            elseif scale <= 1.05 then
                vsr_remove()
                status = string.format("開（不需放大：%d→%d）", h, dh)
            else
                local f = string.format("@%s:d3d11vpp=scale=%.2f:scaling-mode=nvidia", VSR_LABEL, scale)
                status = string.format("開 %.2fx（%d→%d）", scale, h, math.floor(h * scale + 0.5))
                if f == vsr_failed then
                    status = nil
                elseif f ~= vsr_applied then
                    -- a filter with the same label is replaced in place (one reinit, not remove + add)
                    if mp.commandv("vf", "add", f) then
                        vsr_applied, vsr_failed = f, nil
                        msg.info("RTX VSR: " .. f)
                    else
                        vsr_remove(true)
                        vsr_failed = f
                        status = "失敗（d3d11vpp 無法啟用）"
                        if not vsr_warned then
                            vsr_warned = true
                            mp.osd_message("RTX 超解析度無法啟用", 4)
                        end
                    end
                end
            end
        end
    end
    if status and status ~= vsr_status_shown then
        vsr_status_shown = status
        mp.set_property("user-data/oh/vsr-status", status)
    end
end

mp.observe_property("user-data/oh/vsr", "native", function(_, v)
    if type(v) == "string" and v ~= "" then vsr_on = (v == "on") end
    vsr_update()
end)
for _, p in ipairs({ "video-params/h", "display-height", "hwdec-current" }) do
    mp.observe_property(p, "native", vsr_update)
end

-- ---------------------------------------------------------------------------------------------------------------
-- Compare stages. Stages, lowest first: original (pipeline track
-- oh:original), restored (oh:restored, only with RIFE), processed (oh:processed with VSR and smooth playback
-- forced off), vsr (+ RTX VSR), smooth (+ smooth playback). Only stages that exist for the running setup are in the
-- ladder; the top one is the normal view. Hold "o" = lowest stage, hold Shift+o = the stage below the current one,
-- "[" / "]" move the persistent stage. The pipeline lists the tracks it wrote in user-data/oh/compare_layers
-- (comma list of original,restored,processed); empty = compare off. The mpv-side stages (VSR, smooth) still work then.
-- Queue outputs with RIFE (rate_mult 2) store no oh:restored track: when Python publishes
-- user-data/oh/derive-1x = "<video_start>,<src_fps>" the "restored" stage is derived from oh:processed here, as a
-- select of its even slots (the real restored frames; the odd slots are the interpolated ones), see cmp.derive_expr.
-- ---------------------------------------------------------------------------------------------------------------

do
cmp.layers = {}            -- title suffix -> true, from user-data/oh/compare_layers
cmp.derive = nil           -- {start=, period=} from user-data/oh/derive-1x: "restored" = even slots of oh:processed
cmp.graph = nil            -- {n=, k=} while the stream-select graph is installed: all video tracks selected, input k shown
cmp.sticky = nil           -- stage key, nil = top (normal view)
cmp.held = nil             -- stage key shown while a key is down
cmp.hold_key = nil         -- key currently held
cmp.smooth_snap = nil      -- {sync=, interp=, tscale=, on=true}: the user's smooth props while overridden
cmp.label = mp.create_osd_overlay("ass-events")
cmp.watchdog = nil
cmp.applying = false

function cmp.track_id(name)
    for _, t in ipairs(mp.get_property_native("track-list") or {}) do
        if t.type == "video" and t.title == "oh:" .. name then return t.id end
    end
end

-- Switching `vid` deselects the old video track: mpv's demuxer then drops its cached packets (and the new track starts
-- with an empty cache), which on a live tcp stream cannot be refetched: the buffer is gone after every hold of "o". So
-- with two or more compare tracks ALL of them stay selected through one permanent graph
--   [vid1][vid2]...streamselect=inputs=N:map=K[vo]
-- and a stage change only re-sets the map (~2 ms, cache untouched). The graph makes mpv decode in software-copy mode
-- (d3d11va-copy); a vf d3d11vpp (RTX VSR) still works, mpv uploads the frames again. It is installed in the
-- on_preloaded hook, i.e. before mpv selects tracks, so the second track is cached from the first packet on.
function cmp.video_ids()
    local ids = {}
    for _, t in ipairs(mp.get_property_native("track-list") or {}) do
        if t.type == "video" then ids[#ids + 1] = t.id end
    end
    return ids
end

function cmp.vindex(id)       -- 1-based position among the video tracks = vidN of lavfi-complex
    for i, v in ipairs(cmp.video_ids()) do if v == id then return i end end
end

-- user-data/oh/derive-1x "<video_start>,<src_fps>" (fps as a fraction "30000/1001" or a decimal) -> {start, period}
-- period = one slot of the 2x file (1 / (2 * src_fps)) as an expression string
function cmp.parse_derive(v)
    if type(v) ~= "string" then return nil end
    local a, b = v:match("^%s*([^,%s]+)%s*,%s*([^,%s]+)%s*$")
    local start = tonumber(a)
    if not start then return nil end
    local num, den = b:match("^(%d+)/(%d+)$")
    num, den = tonumber(num), tonumber(den)
    local period, pnum
    if num and den and num > 0 and den > 0 then
        period = string.format("%.0f/%.0f", den, 2 * num)
        pnum = den / (2 * num)
    else
        local fps = tonumber(b)
        if not fps or fps <= 0 then return nil end
        period = string.format("%.9f", 1 / (2 * fps))
        pnum = 1 / (2 * fps)
    end
    return { start = start, period = period, pnum = tonumber(pnum) }
end

-- the select expression of the even slots: slot j sits at T0 + j*P (mpv timeline). mpv rebases the timestamps by the
-- demuxer's start time unless rebase-start-time=no (the player's setting): T0 = video_start (- demuxer-start-time).
-- Tolerance is half a slot, so the property being a few ms off does not matter. mod() is floor-based in ffmpeg.
function cmp.derive_t0()
    local d = cmp.derive
    if not d then return nil end
    local t0 = d.start
    if mp.get_property_bool("rebase-start-time", true) then
        t0 = t0 - (mp.get_property_number("demuxer-start-time") or 0)
    end
    return t0
end

function cmp.derive_expr()
    local d = cmp.derive
    if not d then return nil end
    return string.format("eq(mod(floor((t-(%.6f))/(%s)+0.5),2),0)", cmp.derive_t0(), d.period)
end

-- Timestamp snap of the derived layout. The stored timestamps sit on a millisecond grid (Matroska) while the slots do
-- not (P = 1001/60000 s), and the 2x file and the stream-copied 1x original round differently (up to 1 ms apart). The
-- stream-select / overlay frame sync compares timestamps exactly, so such a pair is never "the same instant": extra output
-- frames (measured 72 fps instead of 59.94 on a stream-copy original) and the original one source frame behind. Every
-- input of the graph (oh:processed on the 1x slot grid, a 1x original on the 2x grid) is therefore moved onto
--   T0 + j * P   (j = round((T - T0) / (k * P)) * k, k = 1 for the 2x tracks, 2 for the 1x ones)
-- with ONE expression shape, so equal instants get bit-identical timestamps. k = 1 or 2: slots per frame of the track.
function cmp.snap_expr(k)
    local d = cmp.derive
    if not d then return nil end
    return string.format("setpts='round((%.6f+%d*round((T-(%.6f))/(%d*%s))*%s)/TB)'",
        cmp.derive_t0(), k, cmp.derive_t0(), k, d.period, d.period)
end

-- slots per frame of video track `t` for the snap: 1 = the 2x processed track (and any other 2x track), 2 = a 1x track
function cmp.snap_k(t)
    local d = cmp.derive
    local fps = t and t["demux-fps"]
    if not (d and d.pnum and fps and fps > 0) then return 2 end
    return (fps * d.pnum > 1.5) and 1 or 2
end

-- Seeks are exact (hr-seek): by default mpv's decoders drop every frame before the seek target. That is right for the one
-- displayed track but wrong for a secondary compare track: it must still deliver the frame that precedes the target (a 1x
-- original stays on that frame while the 2x frames in between are shown), else it shows one source frame too old after a
-- paused seek. With several tracks in one graph the drop is therefore switched off (mpv's playloop still drops late
-- frames itself); the user's value is put back when the graph goes away.
function cmp.hr_framedrop(on)
    if on then
        if cmp.hrfd == nil then
            cmp.hrfd = mp.get_property("hr-seek-framedrop") or "yes"
            mp.set_property("hr-seek-framedrop", "no")
        end
    elseif cmp.hrfd ~= nil then
        mp.set_property("hr-seek-framedrop", cmp.hrfd)
        cmp.hrfd = nil
    end
end

-- true: the "restored" layer is derived from oh:processed (derive-1x published, the file has no restored track)
function cmp.derived()
    return cmp.derive ~= nil and cmp.layers.processed == true and cmp.track_id("processed") ~= nil
        and cmp.track_id("restored") == nil
end

-- n = number of video tracks. derive = {p = vid index of oh:processed, expr = select expression}: the derived layer
-- is one more input (the last) and oh:processed goes through a split:
--   [vid1]split[p][x];[x]select='E'[r];[p][vid2][r]streamselect=inputs=3:map=K[vo]
-- (no fps / framestep: streamselect repeats the derived frame at every frame of the 2x inputs)
--   with derive.snap = {[vid index] = setpts filter} (cmp.snap_expr, only when there is a stored original next to the
--   processed track) each input first gets  [vidI]setpts=...[sI];  and the lines above read [sI] for [vidI]
function cmp.graph_str(n, k, derive)
    local ins = {}
    for i = 1, n do ins[i] = "[vid" .. i .. "]" end
    if not derive then
        return string.format("%sstreamselect=inputs=%d:map=%d[vo]", table.concat(ins), n, k)
    end
    local pre, src = "", "[vid" .. derive.p .. "]"
    for i = 1, n do
        local f = derive.snap and derive.snap[i]
        if f then
            pre = pre .. string.format("[vid%d]%s[s%d];", i, f, i)
            ins[i] = "[s" .. i .. "]"
        end
    end
    if derive.snap and derive.snap[derive.p] then src = "[s" .. derive.p .. "]" end
    ins[derive.p] = "[p]"
    ins[n + 1] = "[r]"
    return string.format("%s%ssplit[p][x];[x]select='%s'[r];%sstreamselect=inputs=%d:map=%d[vo]",
        pre, src, derive.expr, table.concat(ins), n + 1, k)
end

-- the snap filters for a graph over the video tracks (nil = no snap: nothing but oh:processed, or no derive info)
function cmp.snap_filters(pid)
    if not cmp.derive then return nil end
    local snap, many = {}, false
    local idx = 0
    for _, t in ipairs(mp.get_property_native("track-list") or {}) do
        if t.type == "video" then
            idx = idx + 1
            snap[idx] = cmp.snap_expr(t.id == pid and 1 or cmp.snap_k(t))
            many = many or t.id ~= pid
        end
    end
    return many and snap or nil
end

-- install / update / remove the graph so that input k (0-based; nil = keep) is shown; true = the graph is active
function cmp.sync_graph(k)
    if cmp.wipe then return cmp.graph ~= nil end
    local n, have = #cmp.video_ids(), 0
    for name in pairs(cmp.layers) do if cmp.track_id(name) then have = have + 1 end end
    local derive
    if cmp.derived() then
        local pid = cmp.track_id("processed")
        derive = { p = cmp.vindex(pid), expr = cmp.derive_expr(), snap = cmp.snap_filters(pid) }
        have = have + 1
    end
    local inputs = n + (derive and 1 or 0)
    if inputs < 2 or have < 2 then
        if cmp.graph then
            cmp.graph = nil
            mp.set_property("lavfi-complex", "")
            mp.set_property("vid", "auto")      -- the graph's going away drops every video selection
        end
        cmp.hr_framedrop(false)
        return false
    end
    k = k or (cmp.graph and cmp.graph.k) or 0
    local key = derive and (derive.expr .. "|" .. table.concat(derive.snap or {}, ";"))
    if cmp.graph and cmp.graph.n == inputs and cmp.graph.k == k and cmp.graph.expr == key then
        return true
    end
    cmp.graph = { n = inputs, k = k, expr = key }
    cmp.hr_framedrop(true)
    mp.set_property("lavfi-complex", cmp.graph_str(n, k, derive))
    return true
end

function cmp.smooth_user()
    if cmp.smooth_snap then return cmp.smooth_snap.on end
    return mp.get_property("video-sync") == "display-resample" or mp.get_property("interpolation") == "yes"
end

function cmp.stages()
    local st, have = {}, {}
    for name in pairs(cmp.layers) do if cmp.track_id(name) then have[name] = true end end
    local derived = cmp.derived()
    if derived then have.restored = true end
    if have.original then st[#st + 1] = { key = "original", name = "原始", track = "original" } end
    if have.restored then st[#st + 1] = { key = "restored", name = "修復", track = "restored", derived = derived } end
    local vsr, smooth = vsr_on, cmp.smooth_user()
    local pname = have.restored and "插幀" or (have.original and "修復" or "基礎")
    local ptrack = have.processed and "processed" or nil
    st[#st + 1] = { key = "processed", name = pname, track = ptrack }
    if vsr then st[#st + 1] = { key = "vsr", name = "+超解析", track = ptrack, vsr = true } end
    if smooth then st[#st + 1] = { key = "smooth", name = "+平滑", track = ptrack, vsr = vsr, smooth = true } end
    return st
end

function cmp.index(st, key)
    for i, s in ipairs(st) do if s.key == key then return i end end
end

-- stage_on: smooth playback should be the user's value; otherwise force it off (snapshotting the user's props)
function cmp.smooth_set(stage_on)
    if stage_on then
        if cmp.smooth_snap then
            local s = cmp.smooth_snap
            cmp.smooth_snap = nil
            mp.set_property("video-sync", s.sync)
            mp.set_property("interpolation", s.interp)
            if s.tscale then mp.set_property("tscale", s.tscale) end
        end
        return
    end
    if cmp.smooth_snap then
        -- Python re-applied the smooth props while overridden (new segment): that is the user's value now
        if mp.get_property("video-sync") == "display-resample" then
            cmp.smooth_snap.sync = "display-resample"
            cmp.smooth_snap.interp = mp.get_property("interpolation")
            cmp.smooth_snap.tscale = mp.get_property("tscale")
        end
    else
        if not cmp.smooth_user() then return end
        cmp.smooth_snap = { on = true, sync = mp.get_property("video-sync"), interp = mp.get_property("interpolation"),
            tscale = mp.get_property("tscale") }
    end
    mp.set_property("video-sync", "audio")
    mp.set_property("interpolation", "no")
end

function cmp.draw_label(stage, top)
    if not stage or stage == top then
        cmp.label:remove()
        return
    end
    cmp.label.res_x, cmp.label.res_y = 1280, 720
    cmp.label.data = string.format("{\\an7\\pos(24,20)\\fs26\\bord2\\1c&HFFFFFF&\\3c&H000000&}%s", ass_esc(stage.name))
    cmp.label:update()
end

-- make the player show stage `key` (nil = top): track, VSR and smooth playback. Idempotent.
-- The stages can have different output sizes (RTX VSR scales the processed stage up to the display height, the lower
-- stages are not scaled; the tracks may also differ in resolution): with mpv's auto-window-resize a stage change would
-- shrink / grow the (windowed) window and put it back on release. So while compare is in use the option is "no": the
-- window keeps its size and every stage is fitted into it (same aspect = same displayed picture). It is locked when a
-- stage is first left (hold / step / wipe) or LOCK_DELAY s after the compare tracks showed up (the first picture and
-- the VSR reconfiguration may still size the window), and the user's value comes back when compare ends.
function cmp.lock_window(on)
    if on then
        if cmp.awr_timer then cmp.awr_timer:kill() cmp.awr_timer = nil end
        if cmp.awr == nil then
            cmp.awr = mp.get_property("auto-window-resize", "yes")
            mp.set_property("auto-window-resize", "no")
        end
    else
        if cmp.awr_timer then cmp.awr_timer:kill() cmp.awr_timer = nil end
        if cmp.awr ~= nil then
            local v = cmp.awr
            cmp.awr = nil
            mp.set_property("auto-window-resize", v)
        end
    end
end

function cmp.lock_update(key)
    if key or cmp.wipe then cmp.lock_window(true)
    elseif next(cmp.layers) == nil then cmp.lock_window(false)
    elseif cmp.awr == nil and not cmp.awr_timer then
        cmp.awr_timer = mp.add_timeout(2, function() cmp.awr_timer = nil cmp.lock_window(true) end)
    end
end

function cmp.apply(key)
    if cmp.applying then return end
    if mp.get_property_native("user-data/oh/passthrough") == "1" then
        -- Keep cmp.layers / sticky as user intent, while ensuring the plain source has no compare graph or stage lock.
        cmp.applying = true
        if cmp.wipe then cmp.wipe.exit(false, true) end
        if cmp.graph then
            cmp.graph = nil
            mp.set_property("lavfi-complex", "")
            mp.set_property("vid", "auto")
        end
        cmp.hr_framedrop(false)
        cmp.lock_window(false)
        cmp.vsr_off = false
        vsr_update()
        cmp.smooth_set(true)
        cmp.label:remove()
        cmp.applying = false
        return
    end
    if cmp.wipe then return end      -- the wipe view (oh_compare_wipe.lua) owns the picture
    cmp.applying = true
    cmp.lock_update(key)
    local st = cmp.stages()
    local i = cmp.index(st, key) or #st
    local stage = st[i]
    local id = stage.track and cmp.track_id(stage.track)
    local idx = id and cmp.vindex(id)
    if stage.derived then idx = #cmp.video_ids() + 1 end       -- the derived layer is the graph's last input
    if not cmp.sync_graph(idx and idx - 1) and id and mp.get_property_number("vid") ~= id then
        mp.set_property_number("vid", id)       -- no graph (single track / not all tracks present): plain track switch
    end
    cmp.vsr_off = (vsr_on and not stage.vsr and i < #st) or false
    vsr_update()
    cmp.smooth_set(stage.smooth == true or i == #st)
    cmp.draw_label(stage, st[#st])
    cmp.applying = false
end

function cmp.effective() return cmp.held or cmp.sticky end

function cmp.release()
    if cmp.watchdog then cmp.watchdog:kill() cmp.watchdog = nil end
    cmp.hold_key = nil
    if cmp.held then
        cmp.held = nil
        cmp.apply(cmp.effective())
    end
end

function cmp.reset()
    cmp.held, cmp.sticky, cmp.hold_key = nil, nil, nil
    if cmp.watchdog then cmp.watchdog:kill() cmp.watchdog = nil end
    cmp.apply(nil)
end

function cmp.hold(key, which)
    if cmp.hold_key == key then return end      -- key repeat
    local st = cmp.stages()
    if #st < 2 then mp.osd_message("對照未啟用", 1.5) return end
    local cur = cmp.index(st, cmp.sticky) or #st
    local i = which == "lowest" and 1 or math.max(1, cur - 1)
    cmp.hold_key, cmp.held = key, st[i].key
    cmp.apply(cmp.held)
    if cmp.watchdog then cmp.watchdog:kill() end
    cmp.watchdog = mp.add_timeout(30, cmp.release)       -- lost key-up
end

function cmp.bind_hold(key, which)
    mp.add_forced_key_binding(key, "oh-cmp-" .. key, function(e)
        if e.event == "down" then cmp.hold(key, which)
        elseif e.event == "up" then if cmp.hold_key == key then cmp.release() end end
    end, { complex = true })
end
cmp.bind_hold("o", "lowest")
cmp.bind_hold("O", "previous")
cmp.bind_hold("Shift+o", "previous")

function cmp.step(d)
    local st = cmp.stages()
    if #st < 2 then mp.osd_message("對照未啟用", 1.5) return end
    local i = math.max(1, math.min(#st, (cmp.index(st, cmp.sticky) or #st) + d))
    cmp.sticky = (i == #st) and nil or st[i].key
    cmp.apply(cmp.effective())
    mp.osd_message(string.format("階段 %d/%d  %s", i, #st, st[i].name), 2)
end
mp.add_forced_key_binding("[", "oh-cmp-down", function() cmp.step(-1) end, { repeatable = true })
mp.add_forced_key_binding("]", "oh-cmp-up", function() cmp.step(1) end, { repeatable = true })

-- a new segment (seek restart, tcp reconnect, append/replace) starts on the default track: drop a hold, re-select
-- (derive-1x is only stored: Python sets it right before compare_layers / the load, which apply it)
mp.observe_property("user-data/oh/derive-1x", "native", function(_, v) cmp.derive = cmp.parse_derive(v) end)
mp.observe_property("user-data/oh/compare_layers", "native", function(_, v)
    -- A source passthrough has no oh:* tracks. Keep the user's compare choice for the restored feeder.
    if (type(v) ~= "string" or v == "")
        and mp.get_property_native("user-data/oh/passthrough") == "1" then return end
    local had = next(cmp.layers) ~= nil
    cmp.layers = {}
    if type(v) == "string" then
        for name in v:gmatch("[^,%s]+") do cmp.layers[(name:gsub("^oh:", ""))] = true end   -- Python publishes the titles ("oh:original")
    end
    if next(cmp.layers) == nil then
        cmp.release()
        if had then cmp.sticky = nil end      -- compare switched off: back to the normal view
    end
    cmp.apply(cmp.effective())
end)
mp.register_event("file-loaded", function()
    cmp.release()
    cmp.apply(cmp.effective())
end)
-- before mpv selects the tracks of a new file: install the graph (or take the old one away) so that every compare track
-- is selected from the first packet on; the layers are read from the property, the observer may not have run yet
mp.add_hook("on_preloaded", 50, function()
    if cmp.wipe then cmp.wipe.exit(false, true) end
    if cmp.watchdog then cmp.watchdog:kill() cmp.watchdog = nil end
    cmp.hold_key, cmp.held = nil, nil
    cmp.derive = cmp.parse_derive(mp.get_property_native("user-data/oh/derive-1x"))
    local v = mp.get_property_native("user-data/oh/compare_layers")
    if type(v) == "string" and (v ~= "" or mp.get_property_native("user-data/oh/passthrough") ~= "1") then
        cmp.layers = {}
        for name in v:gmatch("[^,%s]+") do cmp.layers[(name:gsub("^oh:", ""))] = true end
    elseif mp.get_property_native("user-data/oh/passthrough") ~= "1" then
        cmp.layers = {}
    end
    if cmp.graph then cmp.graph.k = nil end      -- new file: the property still holds the old graph, re-set it
    cmp.apply(cmp.effective())
end)
mp.observe_property("track-list", "native", function()
    if not cmp.applying and (cmp.effective() or next(cmp.layers)) then cmp.apply(cmp.effective()) end
end)
mp.observe_property("focused", "bool", function(_, f) if f == false then cmp.release() end end)
mp.observe_property("idle-active", "bool", function(_, idle) if idle then cmp.reset() end end)
mp.observe_property("user-data/oh/passthrough", "native", function(_, v)
    local on = v == "1"
    if proc.handoff.compare_passthrough and not on then
        proc.handoff.compare_passthrough = false
        -- The source has no compare tracks; reapply the saved intent when the restored clip becomes active.
        mp.add_timeout(0.05, function()
            if mp.get_property_native("user-data/oh/passthrough") ~= "1" then cmp.apply(cmp.effective()) end
        end)
    else
        proc.handoff.compare_passthrough = on
    end
end)
-- Keep a single oh-reset handler: registering again replaces the source reset above
-- (including duration clearing and reset-ack) in mpv.
proc.reset_compare = function() cmp.reset() end
cmp.vsr_update = vsr_update
xpcall(function()   -- wipe view: separate optional file (no top-level locals here: the chunk is near the 200 limit)
    dofile((debug.getinfo(1, "S").source:match("^@?(.*[/\\])") or "") .. "oh_compare_wipe.lua")(cmp)
end, function(e) mp.msg.warn("oh_compare_wipe.lua: " .. tostring(e)) end)
end

-- ---------------------------------------------------------------------------------------------------------------
-- key bindings
-- ---------------------------------------------------------------------------------------------------------------

local steps = {
    LEFT = -10, RIGHT = 10, DOWN = -60, UP = 60, PGDWN = -300, PGUP = 300,
    ["Shift+LEFT"] = -3, ["Shift+RIGHT"] = 3,
    WHEEL_UP = 10, WHEEL_DOWN = -10,
}
for key, delta in pairs(steps) do
    mp.add_forced_key_binding(key, "oh-seek-" .. key, function()
        if key:find("WHEEL") then
            local h = hit()
            if h == "vol" or h == "spk" then change_volume(delta > 0 and 5 or -5) return end
        end
        seek_by(delta)
    end, { repeatable = true })
end
for i = 0, 9 do
    mp.add_forced_key_binding(tostring(i), "oh-jump-" .. i, function() seek_to(duration() * i / 10) end)
end
-- script-message oh-seek <seconds>: an absolute seek exactly like a key press / click (cache check included)
mp.register_script_message("oh-seek", function(t)
    t = tonumber(t)
    if t then seek_to(t) end
end)

-- quality menu: "r" opens it (and moves the highlight while open); mouse: the button left of the fullscreen button
mp.add_forced_key_binding("r", "oh-quality-menu", function()
    if #qualities > 1 then open_menu() end
end)

-- settings menu: "s" (overrides mpv's screenshot key); info panel: "i" (overrides the stats page toggle)
mp.add_forced_key_binding("s", "oh-key-settings", toggle_smenu)
mp.add_forced_key_binding("u", "oh-key-subtitles", toggle_subtitles)
mp.add_forced_key_binding("i", "oh-key-info", toggle_info)
-- add to the processing queue: "a"
mp.add_forced_key_binding("a", "oh-key-queue-add", proc.add)
mp.register_script_message("oh-queue-add", proc.add)
-- cast menu: "c"
mp.add_forced_key_binding("c", "oh-key-cast", toggle_cmenu)

mp.add_forced_key_binding("SPACE", "oh-pause", toggle_pause)
mp.add_forced_key_binding("p", "oh-pause-p", toggle_pause)

-- fullscreen: "f" and double-click are mpv's defaults and stay untouched; Enter and F11 are added
mp.add_forced_key_binding("ENTER", "oh-fs-enter", toggle_fullscreen)
mp.add_forced_key_binding("F11", "oh-fs-f11", toggle_fullscreen)

-- the wheel seeks now; keep volume reachable on the wheel with Ctrl (and on / * as in mpv)
mp.add_forced_key_binding("Ctrl+WHEEL_UP", "oh-vol-up", function() change_volume(2) end, { repeatable = true })
mp.add_forced_key_binding("Ctrl+WHEEL_DOWN", "oh-vol-down", function() change_volume(-2) end, { repeatable = true })
-- volume keys: m = mute, - / = = -5 / +5 % (the volume / mute state is saved by oh_player.py)
mp.add_forced_key_binding("m", "oh-mute", toggle_mute)
mp.add_forced_key_binding("-", "oh-vol-minus", function() change_volume(-5) end, { repeatable = true })
mp.add_forced_key_binding("=", "oh-vol-plus", function() change_volume(5) end, { repeatable = true })
mp.register_script_message("oh-volume", function(v) if tonumber(v) then set_volume(tonumber(v)) end end)
