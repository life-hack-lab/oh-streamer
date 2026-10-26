-- Prefer the concise Chinese AI track once when opening a file. A user's later
-- track switch remains untouched, including disabling subtitles.
local mp = require 'mp'
mp.add_hook('on_preloaded', 60, function()
    if mp.get_property('options/sid') == 'no' then return end
    for _, track in ipairs(mp.get_property_native('track-list', {}) or {}) do
        local lang = (track.lang or ''):lower()
        if track.type == 'sub' and not track.image and
                (lang == 'chi' or lang == 'zho' or lang == 'zh' or lang:match('^zh%-')) and
                (track.title or ''):find('精簡', 1, true) then
            mp.set_property_native('sid', track.id)
            return
        end
    end
end)
