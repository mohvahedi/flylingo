/**
 * Optional bloom, on by default, one prop to switch it off.
 *
 * Why it is here: clearcoat speculars and the emissive connectome glow are the only bright
 * things in an otherwise near black frame, and a small amount of bloom is what makes them
 * read as light rather than as bright paint. It is what turns the fly from "a model lit by
 * a lamp" into "a product shot".
 *
 * Why it is cheap: the mip chain runs at half the drawing buffer (four targets from
 * 1/2 down to 1/16 and back), and only one full resolution composite is added. On a
 * 3070 Ti that is well under a millisecond at 1080p; on a software rasterizer it is the
 * most expensive thing in the scene, which is why it is a prop.
 *
 * Colour management: three only applies tone mapping when it is drawing straight to the
 * canvas, so a pass chain stays in linear HDR and OutputPass applies tone mapping and the
 * output colour space exactly once at the end. No double tone map, no washed out blacks.
 */
import { useEffect, useMemo } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

export type BloomProps = {
  /** bloom contribution, 0 disables the effect without paying for a pass */
  strength?: number;
  /** blur radius across the mip chain */
  radius?: number;
  /** linear HDR luminance above which a pixel blooms. The background sits far below it. */
  threshold?: number;
  /** mip chain resolution as a fraction of the drawing buffer */
  resolutionScale?: number;
};

export function Bloom({
  strength = 0.6,
  radius = 0.45,
  threshold = 0.75,
  resolutionScale = 0.5,
}: BloomProps) {
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const camera = useThree((s) => s.camera);
  const width = useThree((s) => s.size.width);
  const height = useThree((s) => s.size.height);

  const composer = useMemo(() => {
    const c = new EffectComposer(gl);
    c.addPass(new RenderPass(scene, camera));
    c.addPass(
      new UnrealBloomPass(
        new THREE.Vector2(256, 256),
        strength,
        radius,
        threshold,
      ),
    );
    c.addPass(new OutputPass());
    return c;
  }, [gl, scene, camera, strength, radius, threshold]);

  // react-three-fiber sizes the canvas and its pixel ratio for us; mirror both onto the
  // composer's targets, which are separate buffers
  useEffect(() => {
    composer.setPixelRatio(gl.getPixelRatio());
    composer.setSize(width, height);
  }, [composer, gl, width, height]);

  const bloom = useMemo(
    () => composer.passes.find((p): p is UnrealBloomPass => p instanceof UnrealBloomPass) ?? null,
    [composer],
  );
  useEffect(() => {
    if (!bloom) return;
    bloom.setSize(
      Math.max(32, width * resolutionScale),
      Math.max(32, height * resolutionScale),
    );
  }, [bloom, width, height, resolutionScale]);

  useEffect(() => () => composer.dispose(), [composer]);

  // priority > 0 takes the frame over: fiber stops calling gl.render and this renders
  // instead. Pass chain runs after every priority-0 subscriber, so drei's contact shadow
  // pass has already restored its render target by the time we draw.
  useFrame(() => {
    composer.render();
  }, 1);

  return null;
}
