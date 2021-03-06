"""(Very) Simple Implementation of Artnet.

Python Version: 3.6
Source: http://artisticlicence.com/WebSiteMaster/User%20Guides/art-net.pdf
		http://art-net.org.uk/wordpress/structure/streaming-packets/artdmx-packet-definition/

NOTES
- For simplicity: NET and SUBNET not used by default but optional

"""

import socket
from threading import Timer
import time
import math

class StupidArtnet():
	"""(Very) simple implementation of Artnet."""

	UDP_PORT = 6454

	def __init__(self, targetIP='127.0.0.1', universe=0, packet_size=510, fps=30):
		"""Class Initialization."""
		# Instance variables
		self.TARGET_IP = targetIP
		self.SEQUENCE = 0
		self.PHYSICAL = 0
		self.UNIVERSE = universe
		self.SUB = 0
		self.NET = 0
		self.PACKET_SIZE = self.put_in_range(packet_size, 2, 512)
		self.HEADER = bytearray()
		self.BUFFER = bytearray(packet_size)

		self.bIsSimplified = True		# simplify use of universe, net and subnet

		# UDP SOCKET
		self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

		# Timer
		self.fps = fps

		self.make_header()

	def __del__(self):
		self.stop()
		self.close()

	def __str__(self):
		"""Printable object state."""
		s = "===================================\n"
		s += "Stupid Artnet initialized\n"
		s += "Target IP: %s:%i \n" % (self.TARGET_IP, self.UDP_PORT)
		s += "Universe: %i \n" % self.UNIVERSE
		if not (self.bIsSimplified):
			s += "Subnet: %i \n" % self.SUB
			s += "Net: %i \n" % self.NET
		s += "Packet Size: %i \n" % self.PACKET_SIZE
		s += "==================================="

		return s

	def make_header(self):
		"""Make packet header."""
		# 0 - id (7 x bytes + Null)
		self.HEADER = bytearray()
		self.HEADER.extend(bytearray('Art-Net', 'utf8'))
		self.HEADER.append(0x0)
		# 8 - opcode (2 x 8 low byte first)
		self.HEADER.append(0x00)
		self.HEADER.append(0x50)  # ArtDmx data packet
		# 10 - prototocol version (2 x 8 high byte first)
		self.HEADER.append(0x0)
		self.HEADER.append(14)
		# 12 - sequence (int 8), NULL for not implemented
		self.HEADER.append(self.SEQUENCE)
		# 13 - physical port (int 8)
		self.HEADER.append(0x00)
		# 14 - universe, (2 x 8 low byte first)
		if (self.bIsSimplified):
			# not quite correct but good enough for most cases:
			# the whole net subnet is simplified
			# by transforming a single uint16 into its 8 bit parts
			# you will most likely not see any differences in small networks
			v = self.shift_this(self.UNIVERSE)			# convert to MSB / LSB
			self.HEADER.append(v[1])
			self.HEADER.append(v[0])
		# 14 - universe, subnet (2 x 4 bits each)
		# 15 - net (7 bit value)
		else:
			# as specified in Artnet 4 (remember to set the value manually after):
			# Bit 3  - 0 = Universe (1-16)
			# Bit 7  - 4 = Subnet (1-16)
			# Bit 14 - 8 = Net (1-128)
			# Bit 15     = 0
			# this means 16 * 16 * 128 = 32768 universes per port
			# a subnet is a group of 16 Universes
			# 16 subnets will make a net, there are 128 of them
			self.HEADER.append(self.SUB << 4 | self.UNIVERSE)
			self.HEADER.append(self.NET & 0xFF)
		# 16 - packet size (2 x 8 high byte first)
		v = self.shift_this(self.PACKET_SIZE)		# convert to MSB / LSB
		self.HEADER.append(v[0])
		self.HEADER.append(v[1])

	def show(self):
		"""Finally send data."""
		packet = bytearray()
		packet.extend(self.HEADER)
		packet.extend(self.BUFFER)
		try:
			self.s.sendto(packet, (self.TARGET_IP, self.UDP_PORT))
		except Exception as e:
			print("ERROR: Socket error with exception: %s" % e)
		finally:
			self.SEQUENCE = (self.SEQUENCE + 1) % 256

	def close(self):
		"""Close UDP socket."""
		self.s.close()

	##
	# THREADING
	##

	def start(self):
		"""Starts thread clock."""
		self.show()
		self.__clock = Timer((1000.0 / self.fps) / 1000.0, self.start)
		self.__clock.daemon = True
		self.__clock.start()

	def stop(self):
		"""Stops thread clock."""
		self.__clock.cancel()

	##
	# SETTERS - HEADER
	##

	def set_universe(self, universe):
		"""Setter for universe (0 - 15 / 256).

		Mind if protocol has been simplified
		"""
		# This is ugly, trying to keep interface easy
		# With simplified mode the universe will be split into two
		# values, (uni and sub) which is correct anyway. Net will always be 0
		if (self.bIsSimplified):
			self.UNIVERSE = self.put_in_range(universe, 0, 255, False)
		else:
			self.UNIVERSE = self.put_in_range(universe, 0, 15, False)
		self.make_header()

	def set_subnet(self, sub):
		"""Setter for subnet address (0 - 15).

		Set simplify to false to use
		"""
		self.SUB = self.put_in_range(sub, 0, 15, False)
		self.make_header()

	def set_net(self, net):
		"""Setter for net address (0 - 127).

		Set simplify to false to use
		"""
		self.NET = self.put_in_range(net, 0, 127, False)
		self.make_header()

	def set_packet_size(self, packet_size):
		"""Setter for packet size (2 - 512, even only)."""
		self.PACKET_SIZE = self.put_in_range(packet_size, 2, 512, True)
		self.make_header()

	##
	# SETTERS - DATA
	##

	def clear(self):
		"""Clear DMX buffer."""
		self.BUFFER = bytearray(self.PACKET_SIZE)

	def set(self, p):
		"""Set buffer."""
		if len(self.BUFFER) != self.PACKET_SIZE:
			print("ERROR: packet does not match declared packet size")
			return
		self.BUFFER = p

	def set_16bit(self, address, value):
		"""Set single 16bit value in DMX buffer."""
		if address > self.PACKET_SIZE:
			print("ERROR: Address given greater than defined packet size")
			return
		if address < 1 or address > 512 - 1:
			print("ERROR: Address out of range")
			return
		value = self.put_in_range(value, 0, 255, False)
		self.BUFFER[address - 1] = (value) & 0xFF		# low
		self.BUFFER[address] = (value >> 8) & 0xFF		# high

	def set_single_value(self, address, value):
		"""Set single value in DMX buffer."""
		if address > self.PACKET_SIZE:
			print("ERROR: Address given greater than defined packet size")
			return
		if address < 1 or address > 512:
			print("ERROR: Address out of range")
			return
		self.BUFFER[address - 1] = self.put_in_range(value, 0, 255, False)

	def set_single_rem(self, address, value):
		"""Set single value while blacking out others."""
		if address > self.PACKET_SIZE:
			print("ERROR: Address given greater than defined packet size")
			return
		if address < 1 or address > 512:
			print("ERROR: Address out of range")
			return
		self.clear()
		self.BUFFER[address - 1] = self.put_in_range(value, 0, 255, False)

	def set_rgb(self, address, r, g, b):
		"""Set RGB from start address."""
		if address > self.PACKET_SIZE:
			print("ERROR: Address given greater than defined packet size")
			return
		if address < 1 or address > 510:
			print("ERROR: Address out of range")
			return

		self.BUFFER[address - 1] = self.put_in_range(r, 0, 255, False)
		self.BUFFER[address] = self.put_in_range(g, 0, 255, False)
		self.BUFFER[address + 1] = self.put_in_range(b, 0, 255, False)

	##
	# AUX
	##

	def set_simplified(self, bDoSimplify):
		"""Builds Header accordingly.

		True - Header sends 16 bit universe value (OK but incorrect)
		False - Headers sends Universe - Net and Subnet values as protocol
		It is recommended that you set these values with .set_net() and set_physical
		"""
		if (bDoSimplify == self.bIsSimplified):
			return
		self.bIsSimplified = bDoSimplify
		self.make_header()

	def see_header(self):
		"""Show header values."""
		print(self.HEADER)

	def see_buffer(self):
		"""Show buffer values."""
		print(self.BUFFER)

	def blackout(self):
		"""Sends 0's all across."""
		self.clear()
		self.show()

	def flash_all(self):
		"""Sends 255's all across."""
		packet = bytearray(self.PACKET_SIZE)
		[255 for i in packet]
		# for i in range(self.PACKET_SIZE):
		# 	packet[i] = 255
		self.set(packet)
		self.show()

	##
	# UTILS
	##

	@staticmethod
	def shift_this(number, high_first=True):
		"""Utility method: extracts MSB and LSB from number.

		Args:
		number - number to shift
		high_first - MSB or LSB first (true / false)

		Returns:
		(high, low) - tuple with shifted values

		"""
		low = (number & 0xFF)
		high = ((number >> 8) & 0xFF)
		if (high_first):
			return((high, low))
		else:
			return((low, high))
		print("Something went wrong")
		return False

	@staticmethod
	def put_in_range(number, range_min, range_max, make_even=True):
		"""Utility method: sets number in defined range.

		Args:
		number - number to use
		range_min - lowest possible number
		range_max - highest possible number
		make_even - should number be made even

		Returns:
		number - number in correct range

		"""
		if (number < range_min):
			number = range_min
		if (number > range_max):
			number = range_max
		if (make_even and number % 2 != 0):
			number += 1
		return number



class InterspaceArtnet:
    target_ip_head = '2.0.0.'               # typically in 2.x or 10.x range
    first_controller_addres = 10
    packet_size = 510                          # it is not necessary to send whole universe
    last_packet_size = 54
    numController = 3
    numStripesPerController = 8
    numLedsPerStripe = 576
    numStripes = 24
    numUniversesPerController = math.ceil((numStripesPerController * numLedsPerStripe *3) / 510)
    numUniverses = numController * numUniversesPerController
    numLeds = numLedsPerStripe * numStripes
    numPixelPerController = numStripesPerController * numLedsPerStripe
    def __init__(self):
        self.artnet_objects = []
        universe = 0
        for i in range(self.numController):
             for j in range(self.numUniversesPerController):
                ip = self.target_ip_head + str(self.first_controller_addres+i)
                if (i+1) * (j+1) == self.numUniverses:
                   p_size = 54
                else:
                   p_size = self.packet_size
                net = StupidArtnet(ip, universe, p_size)
                self.artnet_objects.append(net)
                universe += 1
    def all_on(self):
        packet = bytearray(self.packet_size)         # create packet for Artnet
        for i in range(self.packet_size):            # fill packet with sequential values
            packet[i] = 100
        for net in self.artnet_objects:
            net.set(packet)
        #self.artnet_objects[3].set(packet)
            #net.flash_all()
        self.show()
    def all_off(self):
        packet = bytearray(self.packet_size)         # create packet for Artnet
        for i in range(self.packet_size):            # fill packet with sequential values
            packet[i] = 0
        for net in self.artnet_objects:
            #net.set(packet)
            net.blackout()
        self.show()
    def send_brightness_buffer(self, led_brightness_buffer):
        universe = 0
        for net in self.artnet_objects:			# is equal to numUniverses
            controllerNumber = int(universe / self.numUniversesPerController)
            pixelOffset = controllerNumber * self.numPixelPerController + (universe - (controllerNumber * self.numUniversesPerController)) * 170
            if (self.numLeds - pixelOffset < 170):
                led_brightness_frame = led_brightness_buffer[pixelOffset:self.numLeds]
            else:
                led_brightness_frame = led_brightness_buffer[pixelOffset:pixelOffset+170]
            #led_brightness_frame = led_brightness_buffer[universe*170:(universe+1)*170]
            buffer = bytearray()
            for brightness in led_brightness_frame:
                buffer.extend([brightness] * 3)	#fill up RGB value with the same monochromic value
            #print(buffer)
            net.set(buffer)
            universe += 1
        self.show()
    def show(self):
        for net in self.artnet_objects:
            net.show()

if __name__ == "__main__":
    inter = InterspaceArtnet()
    FPS = 44

    for i in range(300):

        print("On")
        #inter.all_on()
        numOfLeds = 13824
        theBrightnessBuffer = bytearray(numOfLeds)
        for j in range(numOfLeds):
            if (j < 576):
                theBrightnessBuffer[j] = 70
            else:
                theBrightnessBuffer[j] = 0
        inter.send_brightness_buffer(theBrightnessBuffer)
        time.sleep(1.0/FPS)
        print("Off")
        inter.all_off()
        time.sleep(1.0/FPS)
