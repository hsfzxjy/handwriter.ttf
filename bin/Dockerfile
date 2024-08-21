FROM hsfzxjy/harfbuzz-wasm-fantasy

COPY ./HW_Syn.ttf /usr/share/fonts/HB_WASM_Fantasy.ttf
RUN <<EOF cat >/fantasy/start-gedit.sh
#!/bin/bash
gsettings set org.gnome.gedit.preferences.editor use-default-font 'false'
gsettings set org.gnome.gedit.preferences.editor editor-font 'HB_WASM_Fantasy 60'
gsettings set org.gnome.gedit.preferences.editor display-line-numbers false
gsettings set org.gnome.gedit.preferences.ui statusbar-visible false
gedit
EOF
