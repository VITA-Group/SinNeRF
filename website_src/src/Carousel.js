import { createMedia } from "@artsy/fresnel";
import PropTypes from "prop-types";
import "semantic-ui-css/semantic.css";
import React, { Component } from "react";
import subsection_style from "./style.js";

import {
  Button,
  Container,
  Divider,
  Grid,
  Header,
  Icon,
  Image,
  List,
  Menu,
  Segment,
  Sidebar,
  Visibility,
} from "semantic-ui-react";

import Carousel from "react-multi-carousel";
import "react-multi-carousel/lib/styles.css";

import flower_diet from "./video/flower_diet.mp4";
import flower_dsnerf from "./video/flower_dsnerf.mp4";
import flower_ours from "./video/flower_ours.mp4";
import flower_pix from "./video/flower_pix.mp4";
import fortress_diet from "./video/fortress_diet.mp4";
import fortress_dsnerf from "./video/fortress_dsnerf.mp4";
import fortress_ours from "./video/fortress_ours.mp4";
import fortress_pix from "./video/fortress_pix.mp4";
import hotdog_diet from "./video/hotdog_diet.mp4";
import hotdog_dsnerf from "./video/hotdog_dsnerf.mp4";
import hotdog_ours from "./video/hotdog_ours.mp4";
import hotdog_pix from "./video/hotdog_pix.mp4";
import lego_diet from "./video/lego_diet.mp4";
import lego_dsnerf from "./video/lego_dsnerf.mp4";
import lego_ours from "./video/lego_ours.mp4";
import lego_pix from "./video/lego_pix.mp4";
import room_dietnerf from "./video/room_dietnerf.mp4";
import room_dsnerf from "./video/room_dsnerf.mp4";
import room_ours from "./video/room_ours.mp4";
import room_pix from "./video/room_pix.mp4";

import scan3_ours from "./video/scan3_ours.mp4";
import scan4_ours from "./video/scan4_ours.mp4";
import scan5_ours from "./video/scan5_ours.mp4";
import scan34_ours from "./video/scan34_ours.mp4";
import scan40_ours from "./video/scan40_ours.mp4";
import scan60_ours from "./video/scan60_ours.mp4";
import scan63_ours from "./video/scan63_ours.mp4";
import scan82_ours from "./video/scan82_ours.mp4";
import scan84_ours from "./video/scan84_ours.mp4";

import LazyVideo from "./lazy-video";

const responsive = {
  desktop: {
    breakpoint: { max: 3000, min: 1024 },
    items: 3,
    slidesToSlide: 1, // optional, default to 1.
    partialVisibilityGutter: 40,
  },
  tablet: {
    breakpoint: { max: 1024, min: 464 },
    items: 2,
    slidesToSlide: 1, // optional, default to 1.
    partialVisibilityGutter: 40,
  },
  mobile: {
    breakpoint: { max: 464, min: 0 },
    items: 1,
    slidesToSlide: 1, // optional, default to 1.
    partialVisibilityGutter: 40,
  },
};

const MyCarousel = ({ mobile }) => (
  <div>
    <Carousel
      swipeable={true}
      draggable={true}
      showDots={true}
      responsive={responsive}
      ssr={true} // means to render carousel on server-side.
      infinite={true}
      autoPlay={true}
      autoPlaySpeed={3000}
      keyBoardControl={true}
      containerClass="carousel-container"
      removeArrowOnDeviceType={["tablet", "mobile"]}
      deviceType={mobile}
      dotListClass="custom-dot-list-style"
      itemClass="carousel-item-padding-40-px"
    >
      <LazyVideo no_control src={lego_ours}></LazyVideo>
      <LazyVideo no_control src={hotdog_ours}></LazyVideo>
      <LazyVideo no_control src={flower_ours}></LazyVideo>
      <LazyVideo no_control src={room_ours}></LazyVideo>
      <LazyVideo no_control src={fortress_ours}></LazyVideo>
      <LazyVideo no_control src={scan3_ours}></LazyVideo>
      <LazyVideo no_control src={scan4_ours}></LazyVideo>
      <LazyVideo no_control src={scan5_ours}></LazyVideo>
      <LazyVideo no_control src={scan34_ours}></LazyVideo>
      <LazyVideo no_control src={scan40_ours}></LazyVideo>
      <LazyVideo no_control src={scan60_ours}></LazyVideo>
      <LazyVideo no_control src={scan63_ours}></LazyVideo>
      <LazyVideo no_control src={scan82_ours}></LazyVideo>
      <LazyVideo no_control src={scan84_ours}></LazyVideo>
    </Carousel>
  </div>
);

export default MyCarousel;
