import React from "react";
import LazyLoad from "vanilla-lazyload";
import lazyloadConfig from "./config/lazyload";

// Only initialize it one time for the entire application
if (!document.lazyLoadInstance) {
  document.lazyLoadInstance = new LazyLoad(lazyloadConfig);
}

export class LazyVideo extends React.Component {
  // Update lazyLoad after first rendering of every Video
  componentDidMount() {
    document.lazyLoadInstance.update();
  }

  // Update lazyLoad after rerendering of every Video
  componentDidUpdate() {
    document.lazyLoadInstance.update();
  }

  // Just render the Video with data-src
  render() {
    const { src, no_control } = this.props;
    // const { alt, src, srcset, sizes, width, height } = this.props;
    if (no_control)
      return (
        <video
          src={src}
          className={"lazy"}
          width="100%"
          loop="loop"
          autoplay=""
          muted
          playsInline
        ></video>
      );
    return (
      <video
        src={src}
        className={"lazy"}
        width="100%"
        loop="loop"
        autoplay=""
        controls
        muted
        playsInline
      ></video>
      // <img
      //   alt={alt}
      //   data-src={src}
      //   data-srcset={srcset}
      //   data-sizes={sizes}
      //   width={width}
      //   height={height}
      // />
    );
  }
}

export default LazyVideo;
