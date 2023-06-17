function cosmo --description "cosmo <source.c> [<args>] Compiles using GCC and Cosmopolitan Libc"
  gcc -g -Os -static -nostdlib -nostdinc -fno-pie -no-pie -mno-red-zone -gdwarf-4 \
    -fno-omit-frame-pointer -pg -mnop-mcount -mno-tls-direct-seg-refs \
    -o $argv[1].com.dbg $argv[1].c $argv[2..-1] -fuse-ld=bfd -Wl,-T,ape.lds -Wl,--gc-sections \
    -Icosmo -include cosmo/cosmopolitan.h cosmo/crt.o cosmo/ape-no-modify-self.o cosmo/cosmopolitan.a
  objcopy -S -O binary $argv[1].com.dbg $argv[1].com
end
