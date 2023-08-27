function get_memzoom
  wget -q --no-check-certificate https://storage.googleapis.com/justine/memzoom/memzoom-latest.com
  sudo mv ./memzoom-latest.com /usr/bin/memzoom.com
  sudo chmod +x /usr/bin/memzoom.com
  /usr/bin/memzoom.com -h
end
