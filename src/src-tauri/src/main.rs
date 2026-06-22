// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use tauri::api::process::{Command, CommandEvent};
use tauri::{Manager, State};

/// Shared state for backend connection info
struct AppState {
    api_port: Mutex<u16>,
}

/// Tauri command to get the backend API port
#[tauri::command]
fn get_api_port(state: State<AppState>) -> u16 {
    *state.api_port.lock().unwrap()
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            api_port: Mutex::new(0), // Default to 0 until set
        })
        .invoke_handler(tauri::generate_handler![get_api_port])
        .setup(|app| {
            let app_handle = app.handle();

            // 1. Resolve a writable directory for the user
            // Windows: C:\Users\Name\AppData\Roaming\com.nftool.app\
            let app_data_dir = app
                .path_resolver()
                .app_data_dir()
                .expect("Failed to get app data directory");

            // Create the directory if it doesn't exist
            std::fs::create_dir_all(&app_data_dir)
                .expect("Failed to create app data directory");

            let workspace_path = app_data_dir.to_string_lossy().to_string();

            // 2. Spawn Sidecar
            let (mut rx, _) = Command::new_sidecar("nftool-backend")
                .expect("Failed to create sidecar")
                .args(["--workspace", &workspace_path]) // Pass path to Python
                .spawn()
                .expect("Failed to spawn sidecar");

            // 3. Listen for Handshake
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            // Check for handshake
                            if line.starts_with("NFTOOL_READY:") {
                                if let Some(port_str) = line.strip_prefix("NFTOOL_READY:") {
                                    if let Ok(port) = port_str.trim().parse::<u16>() {
                                        println!("Python Backend Ready on Port: {}", port);

                                        // Update State
                                        let state = app_handle.state::<AppState>();
                                        *state.api_port.lock().unwrap() = port;

                                        // Notify Frontend (Frontend can also poll via get_api_port)
                                        let _ = app_handle.emit_all("backend-ready", port);
                                    }
                                }
                            } else {
                                // Standard Logging
                                println!("PY: {}", line);
                            }
                        }
                        CommandEvent::Stderr(line) => eprintln!("PY ERR: {}", line),
                        CommandEvent::Error(err) => {
                            eprintln!("PY SPAWN ERROR: {}", err);
                        }
                        CommandEvent::Terminated(payload) => {
                            println!(
                                "PY TERMINATED: code={:?}, signal={:?}",
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
