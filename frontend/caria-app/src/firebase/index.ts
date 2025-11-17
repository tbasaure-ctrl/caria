/**
 * Firebase Module - Exportaciones principales
 */

export { app, auth, messaging, analytics, getFCMToken } from './config';
export {
  registerWithEmail,
  loginWithEmail,
  loginWithGoogle,
  logout,
  resetPassword,
  getIdToken,
  onAuthChange,
  getCurrentUser
} from './auth';

