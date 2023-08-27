alias ls="lsd"
alias python="python3"
alias py="python3"
alias wdiff="git diff --no-index --word-diff"
alias cdiff="git diff --no-index --color-words"
# source ~/.poetry/env

function giit
  git commit -am $argv
  git push
end

function fish_title
  echo (status current-command)' '(prompt_pwd)
end

function sunrisesunset
  set DATE (date +%Y-%m-%d)
  set FILE ~/.config/fish/sun/$DATE.txt
  if not test -e $FILE
    set suns ~/.config/fish/sun/*.txt
    if count $suns >/dev/null
      rm -f ~/.config/fish/sun/*.txt
    end
    touch $FILE
    echo (python3 ~/code/py/sunrise/sunrise.py) > $FILE
  end
  cat $FILE
end

function day_suffix
  switch (date +%d)
    case 01 21 31
      echo "st"
    case 2 22
      echo "nd"
    case 3 23
      echo "rd"
    case "*"
      echo "th"
  end
end

set -x fish_greeting "Today is "(date +%A)" the "(date +%d)(day_suffix)" of "(date +%B)" ("(date +%d/%m/%Y)", "(sunrisesunset)", ~"(date +%s)"s unix)"

# # not necessary when using fisher's danhper/fish-ssh-agent plugin, when ssh-add is all we need
# set -x SSH_ENV $HOME/.ssh/environment
# 
# function start_agent
  # echo "Initializing new SSH agent ..."
  # ssh-agent -c | sed 's/^echo/#echo/' > $SSH_ENV
  # chmod 600 $SSH_ENV
  # . $SSH_ENV > /dev/null
  # ssh-add
# end
# 
# function test_identities
    # ssh-add -l | grep "The agent has no identities" > /dev/null
    # if [ $status -eq 0 ]
        # ssh-add
        # if [ $status -eq 2 ]
            # start_agent
        # end
    # end
# end
# 
# if [ -n "$SSH_AGENT_PID" ]
    # ps -ef | grep $SSH_AGENT_PID | grep ssh-agent > /dev/null
    # if [ $status -eq 0 ]
        # test_identities
    # end
# else
    # if [ -f $SSH_ENV ]
        # . $SSH_ENV > /dev/null
    # end
    # ps -ef | grep $SSH_AGENT_PID | grep -v grep | grep ssh-agent > /dev/null
    # if [ $status -eq 0 ]
        # test_identities
    # else
        # start_agent
    # end
# end


# status is-login; and pyenv init --path | source
# pyenv init - | source

starship init fish | source
