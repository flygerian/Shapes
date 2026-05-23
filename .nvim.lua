require("blink.cmp").setup({
	completion = {
		trigger = {
			show_on_keyword = false,
			show_on_trigger_character = false,
		},
	},
	keymap = {
		["<C-Space>"] = { "show", "fallback" },
	},
})

local function run_in_term(cmd, title)
	local buf = vim.api.nvim_create_buf(false, true)

	local width = math.floor(vim.o.columns * 0.8)
	local height = math.floor(vim.o.lines * 0.8)

	vim.api.nvim_open_win(buf, true, {
		relative = "editor",
		width = width,
		height = height,
		row = math.floor((vim.o.lines - height) / 2),
		col = math.floor((vim.o.columns - width) / 2),
		style = "minimal",
		border = "rounded",
		title = " " .. title .. " ",
	})

	local chan = vim.api.nvim_open_term(buf, {})

	local function append(err, data)
		if not data then
			return
		end
		vim.schedule(function()
			vim.api.nvim_chan_send(chan, data)
		end)
	end

	vim.system(cmd, { pty = true, cwd = vim.fn.getcwd(), stdout = append, stderr = append }, function(result)
		vim.schedule(function()
			local win = vim.fn.bufwinid(buf)
			if win ~= -1 then
				vim.api.nvim_win_set_config(win, {
					title = result.code ~= 0 and " " .. title .. " Failed " or " " .. title .. " OK ",
				})
			end
			vim.keymap.set("n", "q", "<cmd>close<CR>", { buffer = buf, silent = true })
		end)
	end)
end

vim.keymap.set("n", "<leader>rb", function()
	run_in_term({ "make", "build-base" }, "Building")
end, { desc = "Shapes: build" })

vim.keymap.set("n", "<leader>ra", function()
	run_in_term({
		"sh",
		"-c",
		"cmake -S base -B base/build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=OFF && make -C base/build && ./base/build/shapes",
	}, "Running")
end, { desc = "Shapes: build and run" })

vim.keymap.set("n", "<leader>rt", function()
	run_in_term({
		"sh",
		"-c",
		"cmake -S base -B base/build -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON && make -C base/build && ./base/build/shapes_test",
	}, "Testing")
end, { desc = "Shapes: build and test" })

-- luasnip
local ls = require("luasnip")
local s = ls.snippet
local i = ls.insert_node
local t = ls.text_node

ls.add_snippets("c", {
	s("case", {
		t("case "),
		i(1, "OP_NAME"),
		t({ ":", "  " }),
		i(2, "fn"),
		t("("),
		i(3),
		t({ ");", "  break;" }),
	}),
})
local ls = require("luasnip")
local s = ls.snippet
local i = ls.insert_node
local t = ls.text_node

ls.add_snippets("c", {
	s("case", {
		t("case "),
		i(1, "OP_NAME"),
		t({ ":", "  " }),
		i(2, "fn"),
		t("("),
		i(3),
		t({ ");", "  break;" }),
	}),
})

--- Perfanno
---

local perfanno = require("perfanno")
local util = require("perfanno.util")

perfanno.setup({
	-- Creates a 10-step RGB color gradient beween background color and "#CC3300"
	line_highlights = util.make_bg_highlights(nil, "#CC3300", 10),
	vt_highlight = util.make_fg_highlight("#CC3300"),
})

local keymap = vim.api.nvim_set_keymap
local opts = { noremap = true, silent = true }

keymap("n", "<LEADER>plf", ":PerfLoadFlat<CR>", opts)
keymap("n", "<LEADER>plg", ":PerfLoadCallGraph<CR>", opts)
keymap("n", "<LEADER>plo", ":PerfLoadFlameGraph<CR>", opts)

keymap("n", "<LEADER>pe", ":PerfPickEvent<CR>", opts)

keymap("n", "<LEADER>pa", ":PerfAnnotate<CR>", opts)
keymap("n", "<LEADER>pf", ":PerfAnnotateFunction<CR>", opts)
keymap("v", "<LEADER>pa", ":PerfAnnotateSelection<CR>", opts)

keymap("n", "<LEADER>pt", ":PerfToggleAnnotations<CR>", opts)

keymap("n", "<LEADER>ph", ":PerfHottestLines<CR>", opts)
keymap("n", "<LEADER>ps", ":PerfHottestSymbols<CR>", opts)
keymap("n", "<LEADER>pc", ":PerfHottestCallersFunction<CR>", opts)
keymap("v", "<LEADER>pc", ":PerfHottestCallersSelection<CR>", opts)
