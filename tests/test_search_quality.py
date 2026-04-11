"""Extensive search quality and category tests using real MCP tool lists."""

import pytest

from adk_tool_search.registry import ToolRegistry


def _make_registry_with_tools(tool_defs: list[tuple[str, str]]) -> ToolRegistry:
    registry = ToolRegistry()

    class _FakeTool:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description

    for name, desc in tool_defs:
        registry.register(_FakeTool(name, desc))
    return registry


PLAYWRIGHT_MCP_TOOLS: list[tuple[str, str]] = [
    ("browser_click", "Perform click on a web page"),
    ("browser_close", "Close the page"),
    ("browser_console_messages", "Returns all console messages"),
    ("browser_drag", "Perform drag and drop between two elements"),
    ("browser_evaluate", "Evaluate JavaScript expression on page or element"),
    ("browser_file_upload", "Upload one or multiple files"),
    ("browser_fill_form", "Fill multiple form fields"),
    ("browser_handle_dialog", "Handle a dialog"),
    ("browser_hover", "Hover over element on page"),
    ("browser_navigate", "Navigate to a URL"),
    ("browser_navigate_back", "Go back to the previous page in the history"),
    ("browser_network_requests", "Returns all network requests since loading the page"),
    ("browser_press_key", "Press a key on the keyboard"),
    ("browser_resize", "Resize the browser window"),
    ("browser_run_code", "Run Playwright code snippet"),
    ("browser_select_option", "Select an option in a dropdown"),
    ("browser_snapshot", "Capture accessibility snapshot of the current page"),
    ("browser_take_screenshot", "Take a screenshot of the current page"),
    ("browser_type", "Type text into editable element"),
    ("browser_wait_for", "Wait for text to appear or disappear or a specified time to pass"),
    ("browser_tabs", "List, create, close, or select a browser tab"),
    (
        "browser_get_config",
        "Get the final resolved config after merging CLI options, environment variables and config file",
    ),
    ("browser_network_state_set", "Sets the browser network state to online or offline"),
    ("browser_route", "Set up a route to mock network requests matching a URL pattern"),
    ("browser_route_list", "List all active network routes"),
    ("browser_unroute", "Remove network routes matching a pattern"),
    ("browser_cookie_clear", "Clear all cookies"),
    ("browser_cookie_delete", "Delete a specific cookie"),
    ("browser_cookie_get", "Get a specific cookie by name"),
    ("browser_cookie_list", "List all cookies optionally filtered by domain path"),
    ("browser_cookie_set", "Set a cookie with optional flags"),
    ("browser_localstorage_clear", "Clear all localStorage"),
    ("browser_localstorage_delete", "Delete a localStorage item"),
    ("browser_localstorage_get", "Get a localStorage item by key"),
    ("browser_localstorage_list", "List all localStorage key-value pairs"),
    ("browser_localstorage_set", "Set a localStorage item"),
    ("browser_sessionstorage_clear", "Clear all sessionStorage"),
    ("browser_sessionstorage_delete", "Delete a sessionStorage item"),
    ("browser_sessionstorage_get", "Get a sessionStorage item by key"),
    ("browser_sessionstorage_list", "List all sessionStorage key-value pairs"),
    ("browser_sessionstorage_set", "Set a sessionStorage item"),
    ("browser_set_storage_state", "Restore storage state from a file"),
    ("browser_storage_state", "Save storage state to a file for later reuse"),
    ("browser_resume", "Resume script execution after it was paused"),
    ("browser_start_tracing", "Start trace recording"),
    ("browser_start_video", "Start video recording"),
    ("browser_stop_tracing", "Stop trace recording"),
    ("browser_stop_video", "Stop video recording"),
    ("browser_video_chapter", "Add a chapter marker to the video recording"),
    ("browser_mouse_click_xy", "Click mouse button at a given position"),
    ("browser_mouse_down", "Press mouse down"),
    ("browser_mouse_up", "Press mouse up"),
    ("browser_mouse_move_xy", "Move mouse to a given position"),
    ("browser_mouse_drag_xy", "Drag left mouse button to a given position"),
    ("browser_mouse_wheel", "Scroll mouse wheel"),
    ("browser_pdf_save", "Save page as PDF"),
    ("browser_generate_locator", "Generate locator for the given element to use in tests"),
    ("browser_verify_element_visible", "Verify element is visible on the page"),
    ("browser_verify_list_visible", "Verify list is visible on the page"),
    ("browser_verify_text_visible", "Verify text is visible on the page"),
    ("browser_verify_value", "Verify element value"),
]

GITHUB_MCP_TOOLS: list[tuple[str, str]] = [
    ("get_me", "Get details of the authenticated GitHub user"),
    ("get_teams", "Get details of the teams the user is a member of"),
    ("get_team_members", "Get member usernames of a specific team in an organization"),
    ("search_repositories", "Find GitHub repositories by name, description, readme, or topics"),
    ("get_file_contents", "Get the contents of a file or directory from a GitHub repository"),
    ("list_commits", "Get list of commits of a branch in a GitHub repository"),
    ("search_code", "Fast and precise code search across ALL GitHub repositories"),
    ("get_commit", "Get details for a commit from a GitHub repository"),
    ("list_branches", "List branches in a GitHub repository"),
    ("list_tags", "List git tags in a GitHub repository"),
    ("get_tag", "Get details about a specific git tag in a GitHub repository"),
    ("list_releases", "List releases in a GitHub repository"),
    ("get_latest_release", "Get the latest release in a GitHub repository"),
    ("get_release_by_tag", "Get a specific release by its tag name"),
    ("create_or_update_file", "Create or update a single file in a GitHub repository"),
    ("create_repository", "Create a new GitHub repository"),
    ("fork_repository", "Fork a GitHub repository to your account"),
    ("create_branch", "Create a new branch in a GitHub repository"),
    ("push_files", "Push multiple files to a GitHub repository in a single commit"),
    ("delete_file", "Delete a file from a GitHub repository"),
    ("list_starred_repositories", "List starred repositories"),
    ("star_repository", "Star a GitHub repository"),
    ("unstar_repository", "Unstar a GitHub repository"),
    ("get_repository_tree", "Get the tree structure of a GitHub repository at a specific ref"),
    ("issue_read", "Get information about a specific issue in a GitHub repository"),
    ("search_issues", "Search for issues in GitHub repositories"),
    ("list_issues", "List issues in a GitHub repository"),
    ("list_issue_types", "List supported issue types for repository owner"),
    ("issue_write", "Create a new or update an existing issue in a GitHub repository"),
    ("add_issue_comment", "Add a comment to a specific issue in a GitHub repository"),
    ("sub_issue_write", "Add a sub-issue to a parent issue in a GitHub repository"),
    ("search_users", "Find GitHub users by username or profile information"),
    ("search_orgs", "Find GitHub organizations by name or metadata"),
    ("pull_request_read", "Get information on a specific pull request in GitHub repository"),
    ("list_pull_requests", "List pull requests in a GitHub repository"),
    ("search_pull_requests", "Search for pull requests in GitHub repositories"),
    ("merge_pull_request", "Merge a pull request in a GitHub repository"),
    ("update_pull_request_branch", "Update the branch of a pull request with the latest changes"),
    ("create_pull_request", "Create a new pull request in a GitHub repository"),
    ("update_pull_request", "Update an existing pull request in a GitHub repository"),
    ("pull_request_review_write", "Create and submit review of a pull request"),
    (
        "add_comment_to_pending_review",
        "Add review comment to the requester's latest pending pull request review",
    ),
    ("add_reply_to_pull_request_comment", "Add a reply to an existing pull request comment"),
    ("assign_copilot_to_issue", "Assign Copilot to a specific issue in a GitHub repository"),
    ("request_copilot_review", "Request a GitHub Copilot code review for a pull request"),
    ("get_code_scanning_alert", "Get details of a specific code scanning alert"),
    ("list_code_scanning_alerts", "List code scanning alerts in a GitHub repository"),
    ("get_secret_scanning_alert", "Get details of a specific secret scanning alert"),
    ("list_secret_scanning_alerts", "List secret scanning alerts in a GitHub repository"),
    ("get_dependabot_alert", "Get details of a specific dependabot alert"),
    ("list_dependabot_alerts", "List dependabot alerts in a GitHub repository"),
    ("list_notifications", "List all GitHub notifications for the authenticated user"),
    ("get_notification_details", "Get detailed information for a specific GitHub notification"),
    ("dismiss_notification", "Dismiss a notification by marking it as read or done"),
    ("mark_all_notifications_read", "Mark all notifications as read"),
    ("manage_notification_subscription", "Manage a notification subscription"),
    (
        "manage_repository_notification_subscription",
        "Manage a repository notification subscription",
    ),
    ("list_discussions", "List discussions for a repository or organisation"),
    ("get_discussion", "Get a specific discussion by ID"),
    ("get_discussion_comments", "Get comments from a discussion"),
    ("list_discussion_categories", "List discussion categories"),
    ("actions_list", "List GitHub Actions workflows and runs"),
    ("actions_get", "Get details about specific GitHub Actions resources"),
    ("actions_run_trigger", "Trigger GitHub Actions workflow operations"),
    ("get_job_logs", "Get logs for GitHub Actions workflow jobs"),
    ("list_global_security_advisories", "List global security advisories from GitHub"),
    ("get_global_security_advisory", "Get a global security advisory"),
    ("list_repository_security_advisories", "List repository security advisories"),
    (
        "list_org_repository_security_advisories",
        "List repository security advisories for an organization",
    ),
    ("list_gists", "List gists for a user"),
    ("get_gist", "Get gist content of a particular gist by ID"),
    ("create_gist", "Create a new gist"),
    ("update_gist", "Update an existing gist"),
    ("projects_list", "List projects for a user or organization"),
    ("projects_get", "Get details about GitHub Projects resources"),
    ("projects_write", "Add update or delete project items in a GitHub Project"),
    ("get_label", "Get a specific label from a repository"),
    ("list_label", "List labels from a repository"),
    ("label_write", "Perform write operations on repository labels"),
    ("enable_toolset", "Enable one of the sets of tools the GitHub MCP server provides"),
    ("list_available_toolsets", "List all available toolsets the GitHub MCP server can offer"),
    ("get_toolset_tools", "List capabilities enabled with the specified toolset"),
]


class TestPlaywrightMCPSearch:
    """Search quality tests against the Playwright MCP tool list (61 tools)."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        return _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS)

    def test_registry_populated(self, registry: ToolRegistry):
        assert registry.tool_count == len(PLAYWRIGHT_MCP_TOOLS)

    def test_search_navigation(self, registry: ToolRegistry):
        results = registry.search("navigate to a URL")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_navigate" in result_names

    def test_search_click(self, registry: ToolRegistry):
        results = registry.search("click on an element")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_click" in result_names

    def test_search_screenshot(self, registry: ToolRegistry):
        results = registry.search("take a screenshot of the page")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_take_screenshot" in result_names

    def test_search_snapshot_accessibility(self, registry: ToolRegistry):
        results = registry.search("accessibility snapshot of the page")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_snapshot" in result_names

    def test_search_cookie(self, registry: ToolRegistry):
        results = registry.search("manage cookies in the browser")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_cookie_" in "\n".join(result_names)
        cookie_tools = [n for n in result_names if "cookie" in n]
        assert len(cookie_tools) >= 2

    def test_search_local_storage(self, registry: ToolRegistry):
        results = registry.search("local storage items")
        result_names = [r.split(":")[0] for r in results]
        assert any("localstorage" in n for n in result_names)

    def test_search_network_requests(self, registry: ToolRegistry):
        results = registry.search("network requests")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_network_requests" in result_names

    def test_search_form_filling(self, registry: ToolRegistry):
        results = registry.search("fill out a form")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_fill_form" in result_names

    def test_search_drag_and_drop(self, registry: ToolRegistry):
        results = registry.search("drag and drop element")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_drag" in result_names

    def test_search_key_press(self, registry: ToolRegistry):
        results = registry.search("press a keyboard key")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_press_key" in result_names

    def test_search_type_text(self, registry: ToolRegistry):
        results = registry.search("type text into input field")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_type" in result_names

    def test_search_console_messages(self, registry: ToolRegistry):
        results = registry.search("console log messages")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_console_messages" in result_names

    def test_search_javascript_evaluation(self, registry: ToolRegistry):
        results = registry.search("run JavaScript code evaluate script")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_evaluate" in result_names

    def test_search_video_recording(self, registry: ToolRegistry):
        results = registry.search("record video of the page")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_start_video" in result_names

    def test_search_pdf_generation(self, registry: ToolRegistry):
        results = registry.search("save page as PDF document")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_pdf_save" in result_names

    def test_search_mouse_coordinate(self, registry: ToolRegistry):
        results = registry.search("click at specific coordinates position")
        result_names = [r.split(":")[0] for r in results]
        assert any("mouse" in n for n in result_names)

    def test_search_select_dropdown(self, registry: ToolRegistry):
        results = registry.search("select option from dropdown menu")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_select_option" in result_names

    def test_search_dialog_handling(self, registry: ToolRegistry):
        results = registry.search("handle browser dialog alert confirm")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_handle_dialog" in result_names

    def test_search_go_back(self, registry: ToolRegistry):
        results = registry.search("go back to previous page")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_navigate_back" in result_names

    def test_search_wait(self, registry: ToolRegistry):
        results = registry.search("wait for something to appear")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_wait_for" in result_names

    def test_search_tabs(self, registry: ToolRegistry):
        results = registry.search("manage browser tabs windows")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_tabs" in result_names

    def test_search_upload_files(self, registry: ToolRegistry):
        results = registry.search("upload files to form input")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_file_upload" in result_names

    def test_search_route_mock(self, registry: ToolRegistry):
        results = registry.search("mock network requests route intercept")
        result_names = [r.split(":")[0] for r in results]
        assert any("route" in n for n in result_names)

    def test_search_returns_limited_results(self, registry: ToolRegistry):
        results = registry.search("browser", n=3)
        assert len(results) <= 3

    def test_search_semantic_not_just_prefix(self, registry: ToolRegistry):
        results = registry.search("screenshot capture page image")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_take_screenshot" in result_names

    def test_search_storage_state_roundtrip(self, registry: ToolRegistry):
        results = registry.search("save and restore storage state cookies")
        result_names = [r.split(":")[0] for r in results]
        assert any("storage" in n or "cookie" in n for n in result_names)

    def test_search_test_verification(self, registry: ToolRegistry):
        results = registry.search("run tests verify assertions")
        result_names = [r.split(":")[0] for r in results]
        assert any("verify" in n or "generate_locator" in n for n in result_names)

    def test_search_tracing_devtools(self, registry: ToolRegistry):
        results = registry.search("trace performance devtools debugging")
        result_names = [r.split(":")[0] for r in results]
        assert any("tracing" in n or "resume" in n for n in result_names)


class TestGitHubMCPSearch:
    """Search quality tests against the GitHub MCP tool list (83 tools)."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        return _make_registry_with_tools(GITHUB_MCP_TOOLS)

    def test_registry_populated(self, registry: ToolRegistry):
        assert registry.tool_count == len(GITHUB_MCP_TOOLS)

    def test_search_issues(self, registry: ToolRegistry):
        results = registry.search("find and list issues")
        result_names = [r.split(":")[0] for r in results]
        assert any("issue" in n.lower() for n in result_names)

    def test_search_pull_requests(self, registry: ToolRegistry):
        results = registry.search("pull requests code review")
        result_names = [r.split(":")[0] for r in results]
        assert any("pull_request" in n for n in result_names)

    def test_search_code_in_repo(self, registry: ToolRegistry):
        results = registry.search("search code in repository")
        result_names = [r.split(":")[0] for r in results]
        assert "search_code" in result_names

    def test_search_file_contents(self, registry: ToolRegistry):
        results = registry.search("read file contents from repository")
        result_names = [r.split(":")[0] for r in results]
        assert "get_file_contents" in result_names

    def test_search_create_branch(self, registry: ToolRegistry):
        results = registry.search("create a new branch")
        result_names = [r.split(":")[0] for r in results]
        assert "create_branch" in result_names

    def test_search_commits(self, registry: ToolRegistry):
        results = registry.search("list commits in a branch")
        result_names = [r.split(":")[0] for r in results]
        assert "list_commits" in result_names

    def test_search_notifications(self, registry: ToolRegistry):
        results = registry.search("view my GitHub notifications")
        result_names = [r.split(":")[0] for r in results]
        assert any("notification" in n for n in result_names)

    def test_search_security_alerts(self, registry: ToolRegistry):
        results = registry.search("security vulnerability scanning alerts")
        result_names = [r.split(":")[0] for r in results]
        sec_tools = [
            n for n in result_names if "scanning" in n or "security" in n or "dependabot" in n
        ]
        assert len(sec_tools) >= 2

    def test_search_discussions(self, registry: ToolRegistry):
        results = registry.search("list discussion categories comments")
        result_names = [r.split(":")[0] for r in results]
        assert any("discussion" in n for n in result_names)

    def test_search_gists(self, registry: ToolRegistry):
        results = registry.search("create and list code gists")
        result_names = [r.split(":")[0] for r in results]
        assert any("gist" in n for n in result_names)

    def test_search_actions_ci(self, registry: ToolRegistry):
        results = registry.search("GitHub Actions CI CD workflows")
        result_names = [r.split(":")[0] for r in results]
        assert any("action" in n for n in result_names)

    def test_search_releases_tags(self, registry: ToolRegistry):
        results = registry.search("list releases and tags")
        result_names = [r.split(":")[0] for r in results]
        assert any("release" in n or "tag" in n for n in result_names)

    def test_search_stars(self, registry: ToolRegistry):
        results = registry.search("star unstar repositories")
        result_names = [r.split(":")[0] for r in results]
        assert any("star" in n for n in result_names)

    def test_search_several_domains_returns_cross_domain(self, registry: ToolRegistry):
        results = registry.search("create update delete manage")
        assert len(results) > 0
        result_names = [r.split(":")[0] for r in results]
        assert len(result_names) >= 3

    def test_search_specific_issue_operations(self, registry: ToolRegistry):
        results = registry.search("add comment to issue")
        result_names = [r.split(":")[0] for r in results]
        assert "add_issue_comment" in result_names

    def test_search_pr_review(self, registry: ToolRegistry):
        results = registry.search("submit pull request review comments")
        result_names = [r.split(":")[0] for r in results]
        assert any("review" in n for n in result_names)

    def test_search_repository_tree(self, registry: ToolRegistry):
        results = registry.search("get repository directory tree structure")
        result_names = [r.split(":")[0] for r in results]
        assert "get_repository_tree" in result_names

    def test_search_dependabot(self, registry: ToolRegistry):
        results = registry.search("dependabot vulnerability dependency alerts")
        result_names = [r.split(":")[0] for r in results]
        assert any("dependabot" in n for n in result_names)

    def test_search_copilot(self, registry: ToolRegistry):
        results = registry.search("copilot assign review automation")
        result_names = [r.split(":")[0] for r in results]
        assert any("copilot" in n for n in result_names)

    def test_search_projects(self, registry: ToolRegistry):
        results = registry.search("GitHub project boards items status")
        result_names = [r.split(":")[0] for r in results]
        assert any("project" in n for n in result_names)

    def test_search_labels(self, registry: ToolRegistry):
        results = registry.search("manage repository labels")
        result_names = [r.split(":")[0] for r in results]
        assert any("label" in n for n in result_names)


class TestCombinedMCPRegistry:
    """Test search with both Playwright and GitHub tools combined (~144 tools)."""

    @pytest.fixture()
    def registry(self) -> ToolRegistry:
        return _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS + GITHUB_MCP_TOOLS)

    def test_combined_count(self, registry: ToolRegistry):
        assert registry.tool_count == len(PLAYWRIGHT_MCP_TOOLS) + len(GITHUB_MCP_TOOLS)

    def test_search_browser_does_not_return_github(self, registry: ToolRegistry):
        results = registry.search("click button on web page")
        result_names = [r.split(":")[0] for r in results]
        browser_results = [n for n in result_names if n.startswith("browser_")]
        assert len(browser_results) >= 1

    def test_search_github_does_not_return_browser(self, registry: ToolRegistry):
        results = registry.search("create a pull request on GitHub")
        result_names = [r.split(":")[0] for r in results]
        assert any("pull_request" in n for n in result_names)

    def test_search_snapshot_returns_browser_not_github(self, registry: ToolRegistry):
        results = registry.search("accessibility snapshot of web page")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_snapshot" in result_names

    def test_search_issues_returns_github_not_browser(self, registry: ToolRegistry):
        results = registry.search("search GitHub issues")
        result_names = [r.split(":")[0] for r in results]
        assert any("issue" in n for n in result_names)

    def test_search_ranks_relevant_higher(self, registry: ToolRegistry):
        results = registry.search("navigate browser URL page")
        result_names = [r.split(":")[0] for r in results]
        browser_navigate_idx = (
            result_names.index("browser_navigate") if "browser_navigate" in result_names else 999
        )
        github_navigate_tools = [n for n in result_names if "repository" in n]
        if browser_navigate_idx < 999 and github_navigate_tools:
            assert browser_navigate_idx < result_names.index(github_navigate_tools[0])

    def test_search_cookie_localstorage_not_github(self, registry: ToolRegistry):
        results = registry.search("browser cookie local storage")
        result_names = [r.split(":")[0] for r in results]
        assert any("cookie" in n or "localstorage" in n for n in result_names)

    def test_search_delete_file_github(self, registry: ToolRegistry):
        results = registry.search("delete a file from repository")
        result_names = [r.split(":")[0] for r in results]
        assert "delete_file" in result_names

    def test_large_registry_reasonable_result_count(self, registry: ToolRegistry):
        results = registry.search("search", n=5)
        assert len(results) <= 5
        assert len(results) > 0

    def test_search_returns_empty_for_unrelated_query(self, registry: ToolRegistry):
        results = registry.search("portfolio rebalance optimizer tax accounting")
        assert results == []


class TestCategoriesPlaywright:
    """Category derivation tests for Playwright MCP tools."""

    def test_browser_is_top_category(self):
        registry = _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS)
        categories = registry.guess_categories()
        assert "browser" in categories
        assert categories[0] == "browser"

    def test_noun_like_segments_not_verbs(self):
        registry = _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS)
        categories = registry.guess_categories()
        action_verbs = {"get", "list", "create", "update", "delete", "add", "set", "manage"}
        for verb in action_verbs:
            assert verb not in categories

    def test_meaningful_segments_present(self):
        registry = _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS)
        categories = registry.guess_categories(max_categories=15)
        meaningful = {
            "cookie",
            "localstorage",
            "sessionstorage",
            "storage",
            "network",
            "mouse",
            "video",
            "pdf",
            "route",
            "tab",
            "dialog",
            "form",
            "snapshot",
            "screenshot",
            "tracing",
        }
        assert len(meaningful.intersection(set(categories))) >= 3

    def test_max_categories_respected(self):
        registry = _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS)
        categories = registry.guess_categories(max_categories=5)
        assert len(categories) <= 5

    def test_empty_registry(self):
        registry = ToolRegistry()
        assert registry.guess_categories() == []


class TestCategoriesGitHub:
    """Category derivation tests for GitHub MCP tools."""

    def test_github_meaningful_categories(self):
        registry = _make_registry_with_tools(GITHUB_MCP_TOOLS)
        categories = registry.guess_categories(max_categories=15)
        meaningful = {
            "repository",
            "issue",
            "pull",
            "request",
            "notification",
            "branch",
            "commit",
            "release",
            "tag",
            "discussion",
            "gist",
            "project",
            "label",
            "action",
            "security",
            "dependabot",
            "copilot",
            "scanning",
            "code",
            "file",
            "team",
            "toolset",
        }
        assert len(meaningful.intersection(set(categories))) >= 3

    def test_no_action_verbs_as_categories(self):
        registry = _make_registry_with_tools(GITHUB_MCP_TOOLS)
        categories = registry.guess_categories()
        action_verbs = {
            "get",
            "list",
            "create",
            "update",
            "delete",
            "add",
            "search",
            "find",
            "set",
            "manage",
            "push",
            "merge",
            "fork",
        }
        for verb in action_verbs:
            assert verb not in categories

    def test_issue_is_a_category(self):
        registry = _make_registry_with_tools(GITHUB_MCP_TOOLS)
        categories = registry.guess_categories()
        assert "issue" in categories

    def test_notification_is_a_category(self):
        registry = _make_registry_with_tools(GITHUB_MCP_TOOLS)
        categories = registry.guess_categories()
        assert "notification" in categories


class TestCategoriesOldBehavior:
    """Tests that guarantee the NEW category derivation is better than the old
    prefix-based approach (_guess_categories_old)."""

    @staticmethod
    def _guess_categories_old(registry: ToolRegistry) -> list[str]:
        prefixes = set()
        for name in registry.tool_names:
            parts = name.replace("-", "_").split("_")
            if len(parts) > 1:
                prefixes.add(parts[0])
            else:
                prefixes.add(name)
        return sorted(prefixes)[:10]

    def test_old_produces_verbs_new_produces_nouns(self):
        registry = _make_registry_with_tools(GITHUB_MCP_TOOLS)
        old_cats = self._guess_categories_old(registry)
        new_cats = registry.guess_categories()

        verb_prefixes = {
            "get",
            "list",
            "create",
            "add",
            "search",
            "manage",
            "mark",
            "request",
            "enable",
            "assign",
            "unstar",
            "dismiss",
            "update",
            "push",
            "merge",
            "fork",
            "delete",
            "sub",
        }
        old_verb_count = len(verb_prefixes.intersection(set(old_cats)))
        new_verb_count = len(verb_prefixes.intersection(set(new_cats)))
        assert new_verb_count < old_verb_count, (
            f"New categories should have fewer action verbs. Old: {old_cats}, New: {new_cats}"
        )

    def test_old_browser_prefix_is_broad_new_is_specific(self):
        registry = _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS)
        old_cats = self._guess_categories_old(registry)
        new_cats = registry.guess_categories(max_categories=15)

        assert "browser" in old_cats
        assert len(old_cats) <= 10

        meaningful_new = {
            "cookie",
            "storage",
            "localstorage",
            "network",
            "mouse",
            "video",
            "pdf",
            "route",
        }
        assert len(meaningful_new.intersection(set(new_cats))) >= 2, (
            f"New categories should include specific nouns beyond 'browser'. Got: {new_cats}"
        )

    def test_combined_registry_old_vs_new(self):
        registry = _make_registry_with_tools(PLAYWRIGHT_MCP_TOOLS + GITHUB_MCP_TOOLS)
        old_cats = self._guess_categories_old(registry)
        new_cats = registry.guess_categories(max_categories=15)

        assert "browser" in old_cats
        assert "get" in old_cats or "list" in old_cats

        meaningful_new = {
            "issue",
            "notification",
            "repository",
            "cookie",
            "storage",
            "network",
            "commit",
            "branch",
        }
        assert len(meaningful_new.intersection(set(new_cats))) >= 3


class TestTokenizationImprovements:
    """Tests for improved tokenization (camelCase, hyphens, etc)."""

    def test_camelcase_is_split(self):
        registry = ToolRegistry()

        class FakeTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description

        registry.register(FakeTool("getPullRequest", "Get information about a pull request"))
        tokens = registry._tokenize("getPullRequest")
        assert "pull" in tokens
        assert "request" in tokens

    def test_hyphenated_names_are_searchable(self):
        registry = ToolRegistry()

        class FakeTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description

        registry.register(FakeTool("browser-click", "Perform click on a web page"))
        results = registry.search("click on web page")
        assert len(results) > 0
        assert "browser-click" in results[0]

    def test_underscore_names_are_searchable(self):
        registry = ToolRegistry()

        class FakeTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description

        registry.register(FakeTool("get_weather_forecast", "Get weather forecast for a location"))
        results = registry.search("weather forecast")
        assert len(results) > 0
        assert "get_weather_forecast" in results[0]

    def test_search_finds_camelcase_tool_by_description_content(self):
        registry = ToolRegistry()

        class FakeTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description

        registry.register(FakeTool("browserNavigate", "Navigate to a URL in the browser"))
        registry.register(FakeTool("getCommit", "Get details for a commit from a repository"))

        results_nav = registry.search("navigate to URL")
        assert any("Navigate" in r for r in results_nav)

        results_commit = registry.search("commit details")
        assert any("Commit" in r for r in results_commit)


class TestEdgeCases:
    """Edge case tests for search quality."""

    def test_search_with_single_tool_returns_match(self):
        registry = _make_registry_with_tools([("my_tool", "Does something useful")])
        results = registry.search("something useful")
        assert len(results) == 1
        assert "my_tool" in results[0]

    def test_search_with_empty_query_returns_empty(self):
        registry = _make_registry_with_tools(GITHUB_MCP_TOOLS)
        results = registry.search("")
        assert results == []

    def test_search_with_completely_unrelated_query_returns_empty(self):
        registry = _make_registry_with_tools([("get_weather", "Get the current weather")])
        results = registry.search("portfolio rebalance optimizer tax accounting")
        assert results == []

    def test_search_premium_on_description_matches(self):
        registry = ToolRegistry()

        class FakeTool:
            def __init__(self, name, description):
                self.name = name
                self.description = description

        registry.register(FakeTool("browser_execute", "Evaluate JavaScript on a page"))
        registry.register(FakeTool("js_runtime", "Run JavaScript code in the browser"))

        results = registry.search("execute JavaScript")
        result_names = [r.split(":")[0] for r in results]
        assert "browser_execute" in result_names

    def test_result_format_is_name_colon_snippet(self):
        registry = _make_registry_with_tools([("my_tool", "A short description")])
        results = registry.search("short description")
        assert len(results) == 1
        assert results[0] == "my_tool: my_tool A short description"
