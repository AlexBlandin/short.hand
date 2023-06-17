function compare
  wc -w $argv
  git diff --word-diff $argv
  echo ""
end