vim.keymap.set("n", "<leader>mbb", function()
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
		title = " Building... ",
	})

	local function append(data)
		if not data then
			return
		end
		vim.schedule(function()
			local lines = vim.split(data, "\n", { trimempty = true })
			local line_count = vim.api.nvim_buf_line_count(buf)
			vim.api.nvim_buf_set_lines(buf, line_count, -1, false, lines)
		end)
	end

	vim.system(
		{ "make", "build-base" },
		{ text = true, cwd = vim.fn.getcwd(), stdout = append, stderr = append },
		function(result)
			vim.schedule(function()
				local win = vim.fn.bufwinid(buf)
				if win ~= -1 then
					vim.api.nvim_win_set_config(win, {
						title = result.code ~= 0 and " Build Failed " or " Build OK ",
					})
				end
				vim.keymap.set("n", "q", "<cmd>close<CR>", { buffer = buf, silent = true })
			end)
		end
	)
end, { desc = "Shapes: build" })
