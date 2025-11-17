/**
 * Firebase Authentication Helpers
 * 
 * Funciones de ayuda para autenticación con Firebase
 */

import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  onAuthStateChanged,
  User,
  GoogleAuthProvider,
  signInWithPopup,
  sendPasswordResetEmail,
  updateProfile,
  UserCredential
} from 'firebase/auth';
import { auth } from './config';

/**
 * Registrar nuevo usuario con email/password
 */
export async function registerWithEmail(
  email: string,
  password: string,
  displayName?: string
): Promise<UserCredential> {
  const userCredential = await createUserWithEmailAndPassword(auth, email, password);
  
  if (displayName && userCredential.user) {
    await updateProfile(userCredential.user, { displayName });
  }
  
  return userCredential;
}

/**
 * Iniciar sesión con email/password
 */
export async function loginWithEmail(
  email: string,
  password: string
): Promise<UserCredential> {
  return signInWithEmailAndPassword(auth, email, password);
}

/**
 * Iniciar sesión con Google
 */
export async function loginWithGoogle(): Promise<UserCredential> {
  const provider = new GoogleAuthProvider();
  return signInWithPopup(auth, provider);
}

/**
 * Cerrar sesión
 */
export async function logout(): Promise<void> {
  return signOut(auth);
}

/**
 * Enviar email de recuperación de contraseña
 */
export async function resetPassword(email: string): Promise<void> {
  return sendPasswordResetEmail(auth, email);
}

/**
 * Obtener token ID de Firebase (para verificar en backend)
 */
export async function getIdToken(forceRefresh = false): Promise<string | null> {
  const user = auth.currentUser;
  if (!user) return null;
  
  try {
    return await user.getIdToken(forceRefresh);
  } catch (error) {
    console.error('Error obteniendo ID token:', error);
    return null;
  }
}

/**
 * Observar cambios en el estado de autenticación
 */
export function onAuthChange(callback: (user: User | null) => void): () => void {
  return onAuthStateChanged(auth, callback);
}

/**
 * Obtener usuario actual
 */
export function getCurrentUser(): User | null {
  return auth.currentUser;
}

