// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::api::process::{Command, CommandEvent};
use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Spawn the Python Sidecar (backend server)
            let (mut rx, _child) = Command::new_sidecar("nftool-backend")
                .expect("failed to create `nftool-backend` binary command")
                .spawn()
                .expect("Failed to spawn sidecar");

            // Get handle to main window for potential future use
            let _main_window = app.get_window("main").unwrap();

            // Listen to sidecar events (stdout/stderr) for debugging
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            println!("[BACKEND] {}", line);
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("[BACKEND ERROR] {}", line);
                        }
                        CommandEvent::Error(err) => {
                            eprintln!("[BACKEND SPAWN ERROR] {}", err);
                        }
                        CommandEvent::Terminated(payload) => {
                            println!(
                                "[BACKEND TERMINATED] code: {:?}, signal: {:?}",
                                payload.code, payload.signal
                            );
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
