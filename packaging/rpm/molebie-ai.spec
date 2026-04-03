Name:           molebie-ai
Version:        0.1.0
Release:        1%{?dist}
Summary:        Self-hosted privacy-first AI assistant CLI

License:        MIT
URL:            https://molebieai.com
Source0:        molebie-ai

# No build steps — we ship a pre-built PyInstaller binary
%global debug_package %{nil}

%description
Molebie AI is a fully self-hosted AI assistant platform with
local LLM inference, web search, RAG, and voice features.
This package installs the molebie-ai CLI tool.

%install
mkdir -p %{buildroot}/usr/local/bin
cp %{SOURCE0} %{buildroot}/usr/local/bin/molebie-ai
chmod 0755 %{buildroot}/usr/local/bin/molebie-ai

%files
/usr/local/bin/molebie-ai

%post
echo ""
echo "molebie-ai installed successfully!"
echo ""
echo "To complete setup, run:"
echo "  molebie-ai install"
echo ""

%changelog
* Thu Apr 03 2026 Jimmy <jimmy@molebieai.com> - 0.1.0-1
- Initial package
