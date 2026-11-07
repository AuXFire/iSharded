# A Homebrew cask. The release workflow fills in the version and hash, and publishes the result
# beside the package. Homebrew reads casks from a tap, so copy that file into one.
cask "isharded" do
  version "__VERSION__"
  sha256 "__SHA256__"

  url "https://github.com/AuXFire/iSharded/releases/download/v#{version}/iSharded.pkg"
  name "iSharded"
  desc "Run one language model across a decentralized pool of machines"
  homepage "https://github.com/AuXFire/iSharded"

  depends_on arch: :arm64

  pkg "iSharded.pkg"

  uninstall launchctl: "ai.isharded.node",
            pkgutil:   "ai.isharded.app"

  # ~/.isharded/up holds the mesh's keys, so only zap removes it.
  zap trash: "~/.isharded"

  caveats "iSharded installs for your user in ~/.isharded. Stop it with: isharded stop"
end
