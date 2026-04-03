# Homebrew formula for molebie-ai
#
# Installation via tap (until accepted into homebrew-core):
#   brew tap Jimmy6929/molebie-ai https://github.com/Jimmy6929/homebrew-molebie-ai
#   brew install molebie-ai
#
# Or directly:
#   brew install Jimmy6929/molebie-ai/molebie-ai

class MolebieAi < Formula
  include Language::Python::Virtualenv

  desc "Self-hosted privacy-first AI assistant CLI"
  homepage "https://molebieai.com"
  url "https://github.com/Jimmy6929/Molebie_AI/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"  # Update with actual sha256 on release
  license "MIT"
  head "https://github.com/Jimmy6929/Molebie_AI.git", branch: "main"

  depends_on "python@3.12"
  depends_on "node" => :recommended

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install_and_link buildpath

    # Ensure the CLI is linked
    bin.install_symlink libexec / "bin" / "molebie-ai"
  end

  def caveats
    <<~EOS
      To complete setup, run:
        molebie-ai install

      This will configure your inference backend, download models,
      and set up optional features (web search, RAG, voice).

      For more info: https://github.com/Jimmy6929/Molebie_AI
    EOS
  end

  test do
    assert_match "molebie-ai", shell_output("#{bin}/molebie-ai --version")
  end
end
