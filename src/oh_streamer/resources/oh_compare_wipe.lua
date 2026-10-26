-- Wipe (split-screen) comparison, loaded by oh_seek.lua via dofile.
-- Key "w": original on the left, the processed stage on the right (the 1x "restored" track when RIFE made one),
-- joined by a vertical divider that follows the mouse while the left button is held (Ctrl+Left / Ctrl+Right nudge it).
-- Needs a software-copy decoder for lavfi-complex: without the permanent stream-select graph of oh_seek.lua hwdec is
-- forced to auto-copy-safe; VSR is off while the wipe is on; everything is put back on exit (track, VSR, smooth playback through cmp.apply, hwdec).
-- Receives oh_seek.lua's `cmp` table; needs cmp.vsr_update (set by the hook) and cmp.layers/track_id/apply.
-- When the 1x "restored" side is derived from oh:processed (queue outputs, cmp.derived(): the even slots, a select), the
-- main (background) input of the overlays must be oh:processed itself, split: overlay follows its main input, and both the
-- derived branch and a stored original (source frame rate: stream copy / 1x encode) run at 1x timestamps, so with either as
-- main the output would drop to the source frame rate (measured: 29.97 instead of 59.94). The processed picture is fully
-- covered by the two crops, it only supplies the 59.94 frame clock (an older design used "original = main": that
-- holds only for an original track at 2x, which the copy layout no longer has).
return function(cmp)
    local W = { on = false, p = 0.5 }
    local label = mp.create_osd_overlay("ass-events")
    local dragging, timer, pending = false, nil, false

    local function video_index(id)      -- vidN in lavfi-complex counts video tracks in order
        local n = 0
        for _, t in ipairs(mp.get_property_native("track-list") or {}) do
            if t.type == "video" then
                n = n + 1
                if t.id == id then return n, t end
            end
        end
    end

    local function pair()      -- left track id, right track id, right label, right side derived from oh:processed
        if not (cmp.layers.original and (cmp.layers.processed or cmp.layers.restored)) then return end
        local l = cmp.track_id("original")
        local derived = cmp.derived and cmp.derived() or false
        local restored = cmp.layers.restored or derived
        local r = cmp.track_id(derived and "processed" or restored and "restored" or "processed")
        if l and r and l ~= r then return l, r, restored and "修復" or "處理後", derived end
    end

    local function draw()
        label.res_x, label.res_y = 1280, 720
        label.data = string.format("{\\an7\\pos(24,20)\\fs26\\bord2\\1c&HFFFFFF&\\3c&H000000&}原始\n"
            .. "{\\an9\\pos(1256,20)\\fs26\\bord2\\1c&HFFFFFF&\\3c&H000000&}%s", W.rname or "")
        label:update()
    end

    local function apply_graph()
        pending = false
        if not W.on then return end
        local li, lt = video_index(W.l)
        local ri = video_index(W.r)
        if not (li and ri) then return end
        local w = (lt and lt.demux_w) or mp.get_property_number("video-params/w") or mp.get_property_number("width") or 1920
        local x = math.floor(w * W.p / 2) * 2
        x = math.max(2, math.min(w - 2, x))
        local g
        if W.derived then
            local expr = cmp.derive_expr()
            if not expr then return end
            -- both inputs go onto the common timestamp grid first (cmp.snap_expr): the stored ms timestamps of the 2x file and
            -- the 1x original differ by up to 1 ms, which would show the original one source frame behind (overlay syncs exactly)
            local sp, sl = cmp.snap_expr(1), cmp.snap_expr(cmp.snap_k(lt))
            local pre, ps, ls = "", string.format("[vid%d]", ri), string.format("[vid%d]", li)
            if sp and sl then
                pre = string.format("[vid%d]%s[sp];[vid%d]%s[sl];", ri, sp, li, sl)
                ps, ls = "[sp]", "[sl]"
            end
            g = string.format("%s%ssplit[m][x];[x]select='%s'[r];[r]crop=iw-%d:ih:%d:0[rr];%scrop=%d:ih:0:0[l];"
                .. "[m][rr]overlay=%d:0[t1];[t1][l]overlay=0:0[t];[t]drawbox=x=%d:y=0:w=2:h=ih:color=white@0.9:t=fill[vo]",
                pre, ps, expr, x, x, ls, x, x, math.max(0, x - 1))
        else
            g = string.format(
                "[vid%d]crop=%d:ih:0:0[l];[vid%d][l]overlay=0:0[t];[t]drawbox=x=%d:y=0:w=2:h=ih:color=white@0.9:t=fill[vo]",
                li, x, ri, math.max(0, x - 1))
        end
        mp.set_property("lavfi-complex", g)
    end

    local function schedule()      -- each update rebuilds the graph (~33 ms): coalesce to ~15 Hz
        if pending then return end
        pending = true
        if timer then timer:kill() end
        timer = mp.add_timeout(0.066, apply_graph)
    end

    local function set_p(p)
        W.p = math.max(0.02, math.min(0.98, p))
        schedule()
    end

    local function mouse_frac()
        local d = mp.get_property_native("osd-dimensions")
        local m = mp.get_property_native("mouse-pos")
        if not (d and m and m.x) then return end
        local vw = d.w - d.ml - d.mr
        if vw <= 0 then return end
        return (m.x - d.ml) / vw
    end

    local function on_mouse()
        if not (W.on and dragging) then return end
        local f = mouse_frac()
        if f then set_p(f) end
    end

    local function bind()
        mp.add_forced_key_binding("MBTN_LEFT", "oh-wipe-drag", function(e)
            if e.event == "down" then
                dragging = true
                on_mouse()
            elseif e.event == "up" then
                dragging = false
            end
        end, { complex = true })
        mp.add_forced_key_binding("Ctrl+LEFT", "oh-wipe-left", function() set_p(W.p - 0.05) end, { repeatable = true })
        mp.add_forced_key_binding("Ctrl+RIGHT", "oh-wipe-right", function() set_p(W.p + 0.05) end, { repeatable = true })
        mp.add_forced_key_binding("ESC", "oh-wipe-esc", function() W.exit(true) end)
    end

    local function unbind()
        for _, n in ipairs({ "oh-wipe-drag", "oh-wipe-left", "oh-wipe-right", "oh-wipe-esc" }) do
            mp.remove_key_binding(n)
        end
    end

    function W.exit(osd, new_stream)
        if not W.on then return end
        W.on, dragging, pending = false, false, false
        if timer then timer:kill() timer = nil end
        unbind()
        label:remove()
        if cmp.graph then
            cmp.graph.k = nil       -- the stream-select graph of oh_seek.lua comes back (cmp.apply below)
        else
            if cmp.hr_framedrop then cmp.hr_framedrop(false) end
            mp.set_property("lavfi-complex", "")
            mp.set_property("vid", "auto")      -- the option still holds the old id: re-setting it would be a no-op
        end
        if W.hwdec then mp.set_property("hwdec", W.hwdec) end
        local vid = (not new_stream) and W.vid or nil
        W.hwdec, W.l, W.r, W.derived = nil, nil, nil, nil
        cmp.wipe = nil
        cmp.vsr_off = false
        -- mpv drops every video selection while the complex graph goes away and treats that moment as EOF: pick the
        -- entry track again (and let the stage logic redo VSR / smooth playback) only once that has settled
        local function finish()
            if cmp.wipe then return end
            if vid and mp.get_property_number("vid") ~= vid then mp.set_property_number("vid", vid) end
            cmp.apply(cmp.effective())
        end
        finish()
        mp.add_timeout(0.15, finish)
        mp.add_timeout(0.6, finish)
        if osd then mp.osd_message("分割線對照：關", 1.5) end
    end

    function W.enter()
        local l, r, rname, derived = pair()
        if not l then mp.osd_message("分割線對照需要兩條畫面（對照未啟用）", 2) return end
        cmp.release()
        if cmp.lock_window then cmp.lock_window(true) end      -- the window keeps its size while the picture changes
        W.on, W.l, W.r, W.rname, W.derived = true, l, r, rname, derived
        W.vid = mp.get_property_number("vid")
        W.hwdec = (not cmp.graph) and mp.get_property("hwdec") or nil      -- the graph already means copy decoding
        cmp.wipe = W
        cmp.vsr_off = true
        if cmp.vsr_update then cmp.vsr_update() end       -- d3d11vpp (VSR) needs d3d11va: off while hwdec copies
        cmp.label:remove()
        if W.hwdec then mp.set_property("hwdec", "auto-copy-safe") end
        if cmp.hr_framedrop then cmp.hr_framedrop(true) end      -- exact seeks: the 1x original keeps its preceding frame
        apply_graph()
        draw()
        bind()
        mp.osd_message("分割線對照：拖曳滑鼠或 Ctrl+左右移動分割線（超解析暫停，w 離開）", 3)
    end

    function W.toggle()
        if W.on then W.exit(true) else W.enter() end
    end

    mp.add_forced_key_binding("w", "oh-cmp-wipe", W.toggle)
    mp.observe_property("mouse-pos", "native", on_mouse)

    -- anything that replaces the stream or drops a layer ends the wipe
    mp.register_event("file-loaded", function() W.exit(false, true) end)
    mp.observe_property("track-list", "native", function()
        if W.on and not pair() then W.exit() end
    end)
    mp.observe_property("user-data/oh/compare_layers", "native", function()
        if W.on and not pair() then W.exit() end
    end)
    local old_reset = cmp.reset
    cmp.reset = function() W.exit(false, true) old_reset() end
    cmp.wipe_api = W
end
