-- Subtitle combos (字幕組合): choose one generated subtitle combo, or compare several at the same time.
-- A separate mpv script (oh_seek.lua is close to Lua's 200-locals limit). Python (player/subtitle_compare.py) publishes
-- user-data/oh/sub-combos = JSON {title, composite, combos:[{id,label,file,style,color,rgb}], truncated} when the opened
-- local video has <video>.subtitle-combos/manifest.json, and "" otherwise. Without it this script does nothing and binds
-- no key.
--
-- Tracks (re-added on every file-loaded: each Python segment is a new mpv file and drops external subtitles):
--   * "比較模式"  = the composite ASS (one style / band / layer per combo, Style == Name == combo "style")
--   * one track per combo (its zh-TW SRT), titled with the combo label
-- Keys (only bound while combos exist):
--   n          next combo as the single subtitle track (leaves compare mode)
--   N          compare mode on / off (all enabled combos at once, each in its own coloured band)
--   Ctrl+n     compare with mpv's native two tracks (sid + secondary-sid) instead of the composite; exactly 2 enabled combos
--   Ctrl+1..8  enable / disable that combo for compare mode (a disabled band is made transparent through
--              sub-ass-style-overrides "<style>.PrimaryColour=..."; the single-combo choice is not affected)
-- Subtitle drawing stays in mpv: nothing here touches Python's seek / restore protocol.

local mp = require 'mp'
local utils = require 'mp.utils'
local msg = require 'mp.msg'

local MAX_KEYS = 8
local data = nil                 -- parsed user-data/oh/sub-combos
local titles = {}                -- combo index -> unique track title
local mode = "single"            -- "single" | "compare"
local cur = 1                    -- combo shown in single mode
local enabled = {}               -- combo index -> bool (compare mode)
local native2 = false            -- compare through sid + secondary-sid when exactly two combos are enabled
local saved_overrides = nil      -- sub-ass-style-overrides before we touched it
local bound = false

local HIDE_FIELDS = { "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour" }
local HIDDEN = "&HFF000000"

local function nenabled()
    local n = 0
    for i = 1, #data.combos do if enabled[i] then n = n + 1 end end
    return n
end

local function enabled_list()
    local out = {}
    for i = 1, #data.combos do if enabled[i] then out[#out + 1] = i end end
    return out
end

local function find_track(title)
    for _, t in ipairs(mp.get_property_native("track-list") or {}) do
        if t.type == "sub" and t.title == title then return t.id end
    end
end

local function ensure_tracks()
    if not find_track(data.title) then
        mp.commandv("sub-add", data.composite, "auto", data.title)
    end
    for i, c in ipairs(data.combos) do
        if not find_track(titles[i]) then
            mp.commandv("sub-add", c.file, "auto", titles[i])
        end
    end
end

local function set_overrides(list)       -- list of strings, or nil to restore the user's own value
    if list == nil then
        if saved_overrides ~= nil then
            mp.set_property_native("sub-ass-style-overrides", saved_overrides)
            saved_overrides = nil
        end
        return
    end
    if saved_overrides == nil then
        saved_overrides = mp.get_property_native("sub-ass-style-overrides") or {}
    end
    local all = {}
    for _, v in ipairs(saved_overrides) do all[#all + 1] = v end
    for _, v in ipairs(list) do all[#all + 1] = v end
    mp.set_property_native("sub-ass-style-overrides", all)
end

local function hidden_overrides()
    local list = {}
    for i, c in ipairs(data.combos) do
        if not enabled[i] then
            for _, f in ipairs(HIDE_FIELDS) do list[#list + 1] = c.style .. "." .. f .. "=" .. HIDDEN end
        end
    end
    if #list == 0 then return nil end
    return list
end

local function set_sid(prop, id)
    mp.set_property(prop, id and tostring(id) or "no")
end

local function describe()
    if mode == "single" then
        return string.format("字幕組合 %d/%d：%s", cur, #data.combos, data.combos[cur].label)
    end
    local lines = {}
    local n = nenabled()
    local how = (native2 and n == 2) and "雙軌" or "合成"
    lines[1] = string.format("字幕比較模式（%s，%d/%d 組）", how, n, #data.combos)
    for i, c in ipairs(data.combos) do
        lines[#lines + 1] = string.format("%s %d. %s [%s]", enabled[i] and "●" or "○", i, c.label, c.color or "")
    end
    return table.concat(lines, "\n")
end

local function apply(announce)
    if not data then return end
    ensure_tracks()
    if announce then mp.set_property_bool("sub-visibility", true) end   -- a segment reload keeps the user's on / off
    if mode == "single" then
        set_overrides(nil)
        set_sid("secondary-sid", nil)
        set_sid("sid", find_track(titles[cur]))
    else
        local list = enabled_list()
        if native2 and #list == 2 then
            set_overrides(nil)
            set_sid("sid", find_track(titles[list[1]]))
            set_sid("secondary-sid", find_track(titles[list[2]]))
        else
            set_sid("secondary-sid", nil)
            set_overrides(hidden_overrides())
            set_sid("sid", find_track(data.title))
        end
    end
    if announce then mp.osd_message(describe(), mode == "compare" and 5 or 3) end
end

local function next_combo()
    if not data then return end
    if mode == "single" then cur = cur % #data.combos + 1 end
    mode = "single"
    apply(true)
end

local function toggle_compare()
    if not data then return end
    mode = (mode == "compare") and "single" or "compare"
    apply(true)
end

local function toggle_native()
    if not data then return end
    if nenabled() ~= 2 then
        mp.osd_message("雙軌模式需要剛好啟用 2 組（Ctrl+1…8 切換）", 3)
        return
    end
    native2 = not native2
    mode = "compare"
    apply(true)
end

local function toggle_combo(i)
    if not data or i > #data.combos then return end
    if enabled[i] and nenabled() <= 1 then
        mp.osd_message("至少要保留一組", 2)
        return
    end
    enabled[i] = not enabled[i]
    if nenabled() ~= 2 then native2 = false end
    mode = "compare"
    apply(true)
end

local function unbind()
    if not bound then return end
    bound = false
    mp.remove_key_binding("oh-subcombo-next")
    mp.remove_key_binding("oh-subcombo-compare")
    mp.remove_key_binding("oh-subcombo-native")
    for i = 1, MAX_KEYS do mp.remove_key_binding("oh-subcombo-toggle-" .. i) end
end

local function bind()
    if bound then return end
    bound = true
    mp.add_forced_key_binding("n", "oh-subcombo-next", next_combo)
    mp.add_forced_key_binding("N", "oh-subcombo-compare", toggle_compare)
    mp.add_forced_key_binding("Ctrl+n", "oh-subcombo-native", toggle_native)
    for i = 1, MAX_KEYS do
        mp.add_forced_key_binding("Ctrl+" .. i, "oh-subcombo-toggle-" .. i, function() toggle_combo(i) end)
    end
end

local function unique_titles(combos)
    local out, seen = {}, {}
    for i, c in ipairs(combos) do
        local t = c.label ~= "" and c.label or c.id
        if seen[t] or t == data.title then t = t .. " (" .. c.id .. ")" end
        seen[t] = true
        out[i] = t
    end
    return out
end

local function reset(value)
    set_overrides(nil)
    data = nil
    if type(value) == "string" and value ~= "" then
        local ok, parsed = pcall(utils.parse_json, value)
        if ok and type(parsed) == "table" and type(parsed.combos) == "table" and #parsed.combos > 0
            and type(parsed.composite) == "string" then
            data = parsed
        else
            msg.warn("sub-combos: unusable value")
        end
    end
    if not data then unbind() return end
    titles = unique_titles(data.combos)
    mode, cur, native2 = "single", 1, false
    enabled = {}
    for i = 1, #data.combos do enabled[i] = true end
    bind()
    if (mp.get_property_number("playlist-count", 0) or 0) > 0 and mp.get_property("path") then
        apply(false)
        mp.osd_message(string.format("找到 %d 組字幕：n 切換　N 比較　Ctrl+n 雙軌　Ctrl+1…8 開關", #data.combos), 4)
    end
end

mp.observe_property("user-data/oh/sub-combos", "native", function(_, v) reset(v) end)
mp.register_event("file-loaded", function()
    if data then apply(false) end
end)

-- M7: the scheduler grows the SRTs while the video plays. Python rewrites the files and bumps user-data/oh/sub-combos-rev
-- (combo list unchanged); drop our external tracks so apply() re-adds them from the fresh files, keeping mode / selection.
local rev_seen = false
mp.observe_property("user-data/oh/sub-combos-rev", "native", function(_, v)
    if not rev_seen then rev_seen = true return end      -- the initial (current) value is not a change
    if not data or v == nil then return end
    local want = { [data.title] = true }
    for i = 1, #data.combos do want[titles[i]] = true end
    for _, t in ipairs(mp.get_property_native("track-list") or {}) do
        if t.type == "sub" and t.external and want[t.title] then mp.commandv("sub-remove", tostring(t.id)) end
    end
    apply(false)
end)
